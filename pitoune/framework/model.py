from .callbacks import CallbackList, ProgressionCallback
from pitoune import torch_to_numpy, tensors_to_variables
import numpy as np
import torch
from torch.autograd import Variable

class Model(object):
    def __init__(self, model, optimizer, loss_function, metrics=[]):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.metric_names = [metric.__name__ for metric in metrics]

    def fit_generator(self, train_generator, valid_generator, n_epochs=1000, steps_per_epoch=None, callbacks=[]):
        callbacks = [ProgressionCallback()] + callbacks
        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)

        train_steps_per_epoch = self._get_steps_per_epoch(train_generator, steps_per_epoch)
        params = {'n_epochs': n_epochs, 'steps_per_epoch': train_steps_per_epoch, 'metrics': self.metric_names}
        callback_list.set_params(params)

        logs = []
        self.stop_training = False
        callback_list.on_train_begin(logs)
        for epoch in range(1, n_epochs + 1):
            callback_list.on_epoch_begin(epoch, logs)
            logs.append({})
            losses_sum = 0.
            metrics_sum = np.zeros(len(self.metrics))
            times_sum = 0.

            train_iterator = iter(train_generator)
            for step in range(1, train_steps_per_epoch + 1):
                callback_list.on_batch_begin(step, logs)

                self.model.zero_grad()

                loss_tensor, metrics_tensors = self._run_step(train_iterator)

                loss_tensor.backward()
                self.optimizer.step()

                loss = torch_to_numpy(loss_tensor)
                losses_sum += loss
                metrics = np.array(torch_to_numpy(metrics_tensors))
                metrics_sum += metrics

                losses_mean = losses_sum / step
                metrics_mean = metrics_sum / step

                metrics_dict = dict(zip(self.metric_names, metrics_mean))
                logs[-1] = {'epoch': epoch, 'lr': self.optimizer.param_groups[0]['lr'], 'loss': losses_mean, **metrics_dict}
                callback_list.on_batch_end(step, logs)


            val_loss, val_metrics = self._validate(valid_generator, steps_per_epoch)
            val_metrics_dict = {'val_' + metric_name:metric for metric_name, metric in zip(self.metric_names, val_metrics)}

            logs[-1] = {'epoch': epoch, 'lr': self.optimizer.param_groups[0]['lr'], 'loss': losses_mean, **metrics_dict, 'val_loss': val_loss, **val_metrics_dict}
            callback_list.on_epoch_end(epoch, logs)

            if self.stop_training:
                break

        callback_list.on_train_end(logs)

        self.best_epoch = min(logs, key=lambda x: x['val_loss'])

        return logs

    def predict(self, x):
        x = tensors_to_variables(x)
        return self.model(x)

    def _validate(self, valid_generator, steps_per_epoch):
        valid_steps_per_epoch = self._get_steps_per_epoch(valid_generator, steps_per_epoch)
        losses = np.empty(valid_steps_per_epoch)
        metrics_list = np.empty((valid_steps_per_epoch, len(self.metrics)))
        valid_iterator = iter(valid_generator)
        for step in range(valid_steps_per_epoch):
            loss_tensor, metrics_tensors = self._run_step(valid_iterator)
            losses[step] = torch_to_numpy(loss_tensor)
            metrics_list[step] = np.array(torch_to_numpy(metrics_tensors))
        return losses.mean(), metrics_list.mean(0)

    def _get_steps_per_epoch(self, iterator, steps_per_epoch):
        if steps_per_epoch is None:
            steps_per_epoch = len(iterator)
        return steps_per_epoch

    def _run_step(self, iterator):
        x, y = next(iterator)
        x = tensors_to_variables(x)
        y = tensors_to_variables(y)
        pred_y = self.model(x)
        loss_tensor = self.loss_function(pred_y, y)
        metrics = self._compute_metrics(pred_y, y)
        return loss_tensor, metrics

    def _compute_metrics(self, pred_y, y):
        return [metric(pred_y, y) for metric in self.metrics]

    def load_weights(self, filename):
        self.model.load_state_dict(torch.load(filename))

    def save_weights(self, filename):
        torch.save(self.model.state_dict(), filename)

    def get_weights(self):
        return self.model.state_dict()

    def cuda(*args, **kwargs):
        return self.model.cuda(*args, **kwargs)

    def cpu(*args, **kwargs):
        return self.model.cpu(*args, **kwargs)
