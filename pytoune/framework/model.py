from .callbacks import CallbackList, ProgressionCallback
from pytoune import torch_to_numpy, tensors_to_variables, variables_to_tensors
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

class Model:
    def __init__(self, model, optimizer, loss_function, metrics=[]):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.metrics_names = [metric.__name__ for metric in metrics]

    def fit(self, x, y, validation_x=None, validation_y=None, batch_size=32, epochs=1000, steps_per_epoch=None, validation_steps=None, verbose=True, callbacks=[]):
        train_dataset = TensorDataset(x, y)
        train_generator = DataLoader(train_dataset, batch_size)

        valid_generator = None
        if validation_x is not None or validation_y is not None:
            validation_dataset = TensorDataset(validation_x, validation_y)
            valid_generator = DataLoader(validation_dataset, batch_size)

        return self.fit_generator(train_generator,
                                  valid_generator=valid_generator,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps,
                                  verbose=verbose,
                                  callbacks=callbacks)

    def fit_generator(self, train_generator, valid_generator=None, epochs=1000, steps_per_epoch=None, validation_steps=None, verbose=True, callbacks=[]):
        if verbose:
            callbacks = [ProgressionCallback()] + callbacks
        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)

        if validation_steps is None:
            if hasattr(valid_generator, '__len__'):
                validation_steps = len(valid_generator)
            elif steps_per_epoch is not None:
                validation_steps = steps_per_epoch
            else:
                raise ValueError("Invalid 'validation_steps' value. Either a value for 'validation_steps' or 'steps_per_epoch' must be provided, or 'valid_generator' must provide a '__len__' method.")
        if steps_per_epoch is None:
            steps_per_epoch = len(train_generator)
        params = {'epochs': epochs, 'steps': steps_per_epoch}
        callback_list.set_params(params)

        epoch_logs = []
        self.stop_training = False
        callback_list.on_train_begin({})
        for epoch in range(1, epochs + 1):
            callback_list.on_epoch_begin(epoch, {})
            losses_sum = 0.
            metrics_sum = np.zeros(len(self.metrics))
            times_sum = 0.

            self.model.train(True)
            train_iterator = iter(train_generator)
            for step in range(1, steps_per_epoch + 1):
                callback_list.on_batch_begin(step, {})

                self.model.zero_grad()

                x, y = next(train_iterator)
                loss_tensor, metrics_tensors = self._compute_loss_and_metrics(x, y)

                loss_tensor.backward()
                self.optimizer.step()

                loss, metrics = self._loss_and_metrics_tensors_to_numpy(loss_tensor, metrics_tensors)
                losses_sum += loss
                metrics_sum += metrics

                metrics_dict = dict(zip(self.metrics_names, metrics))
                batch_logs = {'batch': step, 'loss': loss, **metrics_dict}
                callback_list.on_batch_end(step, batch_logs)

            val_dict = {}
            if valid_generator is not None:
                self.model.eval()
                val_loss, val_metrics = self._validate(valid_generator, validation_steps)
                val_metrics_dict = {'val_' + metric_name:metric for metric_name, metric in zip(self.metrics_names, val_metrics)}
                val_dict = {'val_loss': val_loss, **val_metrics_dict}

            losses_mean = losses_sum / steps_per_epoch
            metrics_mean = metrics_sum / steps_per_epoch
            metrics_dict = dict(zip(self.metrics_names, metrics_mean))
            epoch_log = {'epoch': epoch, 'loss': losses_mean, **metrics_dict, **val_dict}
            callback_list.on_epoch_end(epoch, epoch_log)

            epoch_logs.append(epoch_log)

            if self.stop_training:
                break

        callback_list.on_train_end({})

        return epoch_logs

    def predict(self, x):
        self.model.eval()
        x = tensors_to_variables(x, volatile=True)
        return variables_to_tensors(self.model(x))

    def evaluate(self, x, y, return_pred=False):
        self.model.eval()
        return variables_to_tensors(self._compute_loss_and_metrics(x, y, return_pred=return_pred))

    def _validate(self, valid_generator, validation_steps):
        losses_list = np.zeros(validation_steps)
        metrics_list = np.zeros((validation_steps,len(self.metrics)))
        valid_iterator = iter(valid_generator)
        for step in range(validation_steps):
            x, y = next(valid_iterator)
            loss_tensor, metrics_tensors = self._compute_loss_and_metrics(x, y)
            loss, metrics = self._loss_and_metrics_tensors_to_numpy(loss_tensor, metrics_tensors)
            losses_list[step] = loss
            metrics_list[step] = metrics
        return losses_list.mean(), metrics_list.mean(0)

    def _compute_loss_and_metrics(self, x, y, return_pred=False):
        x = tensors_to_variables(x, volatile=not self.model.training)
        y = tensors_to_variables(y, volatile=not self.model.training)
        pred_y = self.model(x)
        loss_tensor = self.loss_function(pred_y, y)
        metrics_tensors = self._compute_metrics(pred_y, y)
        ret = (loss_tensor, metrics_tensors)
        if return_pred:
            ret = ret + (pred_y,)
        return ret

    def _compute_metrics(self, pred_y, y):
        return [metric(pred_y, y) for metric in self.metrics]

    def _loss_and_metrics_tensors_to_numpy(self, loss_tensor, metrics_tensors):
        loss = float(loss_tensor)
        metrics = np.array([])
        if len(metrics_tensors) > 0:
            metrics = np.array(torch_to_numpy(metrics_tensors))
            metrics = metrics.squeeze(1)
        return loss, metrics

    def load_weights(self, filename):
        self.set_weights(torch.load(filename))

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), filename)

    def get_weights(self):
        return self.model.state_dict()

    def get_weight_copies(self):
        weights = self.get_weights()
        for k in weights.keys():
            weights[k] = weights[k].cpu().clone()
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def cuda(*args, **kwargs):
        return self.model.cuda(*args, **kwargs)

    def cpu(*args, **kwargs):
        return self.model.cpu(*args, **kwargs)
