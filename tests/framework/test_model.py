import os

import unittest
from unittest import TestCase, skipIf
from unittest.mock import MagicMock, call, ANY

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from poutyne.framework import Model
from poutyne.framework import warning_settings


some_metric_1_value = 1.
some_metric_2_value = 2.

def some_metric_1(y, y_pred):
    # pylint: disable=unused-argument
    return torch.FloatTensor([some_metric_1_value])

def some_metric_2(y, y_pred):
    # pylint: disable=unused-argument
    return torch.FloatTensor([some_metric_2_value])

def some_data_tensor_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y

def some_ndarray_generator(batch_size):
    while True:
        x = np.random.rand(batch_size, 1).astype(np.float32)
        y = np.random.rand(batch_size, 1).astype(np.float32)
        yield x, y

class SomeDataGeneratorUsingStopIteration:
    def __init__(self, batch_size, length):
        self.batch_size = batch_size
        self.length = length

    def __iter__(self):
        return ((np.random.rand(self.batch_size, 1).astype(np.float32),
                 np.random.rand(self.batch_size, 1).astype(np.float32))
                for i in range(self.length))

class SomeDataGeneratorWithLen:
    def __init__(self, batch_size, length, num_missing_samples):
        self.batch_size = batch_size
        self.length = length
        self.num_generator_called = 0
        self.x = torch.rand(length * batch_size - num_missing_samples, 1)
        self.y = torch.rand(length * batch_size - num_missing_samples, 1)

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        start_idx = self.num_generator_called * self.batch_size
        end_idx = (self.num_generator_called + 1) * self.batch_size
        x = self.x[start_idx:end_idx]
        y = self.y[start_idx:end_idx]
        self.num_generator_called += 1
        if self.num_generator_called == self.length:
            self.num_generator_called = 0
        return x, y



class ModelTest(TestCase):
    # pylint: disable=too-many-public-methods
    epochs = 10
    steps_per_epoch = 5
    batch_size = 20

    evaluate_dataset_len = 107

    cuda_device = int(os.environ.get('CUDA_DEVICE', 0))

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_module = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_module.parameters(), lr=1e-3)
        self.metrics = [some_metric_1, some_metric_2]
        self.metrics_names = ['some_metric_1', 'some_metric_2']
        self.metrics_values = [some_metric_1_value, some_metric_2_value]

        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function,
                           metrics=self.metrics)
        self.mock_callback = MagicMock()

    def test_fitting_tensor_generator(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(train_generator, valid_generator,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=ModelTest.steps_per_epoch,
                                        validation_steps=ModelTest.steps_per_epoch,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_fitting(params, logs)

    def test_fitting_without_valid_generator(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(train_generator, None,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=ModelTest.steps_per_epoch,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_fitting(params, logs, has_valid=False)

    def test_fitting_ndarray_generator(self):
        train_generator = some_ndarray_generator(ModelTest.batch_size)
        valid_generator = some_ndarray_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(train_generator, valid_generator,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=ModelTest.steps_per_epoch,
                                        validation_steps=ModelTest.steps_per_epoch,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_fitting(params, logs)

    def test_fitting_with_data_loader(self):
        # pylint: disable=too-many-locals
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - \
                     train_final_batch_missing_samples
        train_x = torch.rand(train_size, 1)
        train_y = torch.rand(train_size, 1)
        train_dataset = TensorDataset(train_x, train_y)
        train_generator = DataLoader(train_dataset, train_batch_size)

        valid_real_steps_per_epoch = 10
        valid_batch_size = 15
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - \
                     valid_final_batch_missing_samples
        valid_x = torch.rand(valid_size, 1)
        valid_y = torch.rand(valid_size, 1)
        valid_dataset = TensorDataset(valid_x, valid_y)
        valid_generator = DataLoader(valid_dataset, valid_batch_size)

        logs = self.model.fit_generator(train_generator, valid_generator,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=None,
                                        validation_steps=None,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs)

    def test_fitting_with_tensor(self):
        # pylint: disable=too-many-locals
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - \
                     train_final_batch_missing_samples
        train_x = torch.rand(train_size, 1)
        train_y = torch.rand(train_size, 1)

        valid_real_steps_per_epoch = 10
        # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_batch_size = train_batch_size
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - \
                     valid_final_batch_missing_samples
        valid_x = torch.rand(valid_size, 1)
        valid_y = torch.rand(valid_size, 1)

        logs = self.model.fit(train_x, train_y,
                              validation_x=valid_x,
                              validation_y=valid_y,
                              epochs=ModelTest.epochs,
                              batch_size=train_batch_size,
                              steps_per_epoch=None,
                              validation_steps=None,
                              callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs)

    def test_fitting_with_np_array(self):
        # pylint: disable=too-many-locals
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - \
                     train_final_batch_missing_samples
        train_x = np.random.rand(train_size, 1).astype(np.float32)
        train_y = np.random.rand(train_size, 1).astype(np.float32)

        valid_real_steps_per_epoch = 10
        # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_batch_size = train_batch_size
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - \
                     valid_final_batch_missing_samples
        valid_x = np.random.rand(valid_size, 1).astype(np.float32)
        valid_y = np.random.rand(valid_size, 1).astype(np.float32)

        logs = self.model.fit(train_x, train_y,
                              validation_x=valid_x,
                              validation_y=valid_y,
                              epochs=ModelTest.epochs,
                              batch_size=train_batch_size,
                              steps_per_epoch=None,
                              validation_steps=None,
                              callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs)

    def test_fitting_with_generator_with_len(self):
        train_real_steps_per_epoch = 30
        train_generator = SomeDataGeneratorWithLen(batch_size=ModelTest.batch_size,
                                                   length=train_real_steps_per_epoch,
                                                   num_missing_samples=7)
        valid_generator = SomeDataGeneratorWithLen(batch_size=15,
                                                   length=10,
                                                   num_missing_samples=3)
        logs = self.model.fit_generator(train_generator, valid_generator,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=None,
                                        validation_steps=None,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs)

    def test_fitting_with_generator_with_stop_iteration(self):
        train_real_steps_per_epoch = 30
        train_generator = SomeDataGeneratorUsingStopIteration(batch_size=ModelTest.batch_size,
                                                              length=train_real_steps_per_epoch)
        valid_generator = SomeDataGeneratorUsingStopIteration(batch_size=15, length=10)
        logs = self.model.fit_generator(train_generator, valid_generator,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=None,
                                        validation_steps=None,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': None}
        self._test_fitting(params, logs, steps=train_real_steps_per_epoch)

    def _test_fitting(self, params, logs, has_valid=True, steps=None):
        if steps is None:
            steps = params['steps']
        self.assertEqual(len(logs), params['epochs'])
        train_dict = dict(zip(self.metrics_names, self.metrics_values), loss=ANY, time=ANY)
        if has_valid:
            val_metrics_names = ['val_' + metric_name for metric_name in self.metrics_names]
            val_dict = dict(zip(val_metrics_names, self.metrics_values), val_loss=ANY)
            log_dict = {**train_dict, **val_dict}
        else:
            log_dict = train_dict

        for epoch, log in enumerate(logs, 1):
            self.assertEqual(log, dict(log_dict, epoch=epoch))

        call_list = []
        call_list.append(call.on_train_begin({}))
        for epoch in range(1, params['epochs']+1):
            call_list.append(call.on_epoch_begin(epoch, {}))
            for step in range(1, steps+1):
                call_list.append(call.on_batch_begin(step, {}))
                call_list.append(call.on_backward_end(step))
                call_list.append(call.on_batch_end(
                    step, {'batch': step, 'size': ANY, **train_dict}
                ))
            call_list.append(call.on_epoch_end(epoch, {'epoch': epoch, **log_dict}))
        call_list.append(call.on_train_end({}))

        method_calls = self.mock_callback.method_calls
        self.assertIn(call.set_model(self.model), method_calls[:2])
        self.assertIn(call.set_params(params), method_calls[:2])

        self.assertEqual(len(method_calls), len(call_list) + 2)
        self.assertEqual(method_calls[2:], call_list)

    def test_tensor_train_on_batch(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        loss, metrics = self.model.train_on_batch(x, y)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_train_on_batch_with_pred(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        loss, metrics, pred_y = self.model.train_on_batch(x, y, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_ndarray_train_on_batch(self):
        x = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        y = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        loss, metrics = self.model.train_on_batch(x, y)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_evaluate(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        loss, metrics = self.model.evaluate(x, y, batch_size=ModelTest.batch_size)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_evaluate_with_pred(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        # We also test the unpacking.
        # pylint: disable=unused-variable
        loss, metrics, pred_y = self.model.evaluate(x, y,
                                                    batch_size=ModelTest.batch_size,
                                                    return_pred=True)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_np_array(self):
        x = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        y = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        loss, metrics, pred_y = self.model.evaluate(x, y,
                                                    batch_size=ModelTest.batch_size,
                                                    return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_data_loader(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        dataset = TensorDataset(x, y)
        generator = DataLoader(dataset, ModelTest.batch_size)
        loss, metrics, pred_y = self.model.evaluate_generator(generator, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])
        self._test_predictions_for_evaluate_and_predict_generator(pred_y)

    def test_evaluate_generator(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        loss, metrics, pred_y = self.model.evaluate_generator(generator,
                                                              steps=num_steps,
                                                              return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])
        for pred in pred_y:
            self.assertEqual(type(pred), np.ndarray)
            self.assertEqual(pred.shape, (ModelTest.batch_size, 1))
        self.assertEqual(np.concatenate(pred_y).shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_with_only_one_metric(self):
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function,
                           metrics=self.metrics[:1])
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        loss, first_metric = self.model.evaluate(x, y, batch_size=ModelTest.batch_size)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(first_metric), float)
        self.assertEqual(first_metric, some_metric_1_value)

    def test_metrics_integration(self):
        num_steps = 10
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function,
                           metrics=[F.mse_loss])
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)
        self.model.fit_generator(train_generator, valid_generator,
                                 epochs=ModelTest.epochs,
                                 steps_per_epoch=ModelTest.steps_per_epoch,
                                 validation_steps=ModelTest.steps_per_epoch,
                                 callbacks=[self.mock_callback])
        generator = some_data_tensor_generator(ModelTest.batch_size)
        loss, mse = self.model.evaluate_generator(generator, steps=num_steps)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(mse), float)

    def test_evaluate_with_no_metric(self):
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function)
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        loss = self.model.evaluate(x, y, batch_size=ModelTest.batch_size)
        self.assertEqual(type(loss), float)

    def test_tensor_evaluate_on_batch(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        loss, metrics = self.model.evaluate_on_batch(x, y)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_evaluate_on_batch_with_pred(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        loss, metrics, pred_y = self.model.evaluate_on_batch(x, y, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_ndarray_evaluate_on_batch(self):
        x = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        y = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        loss, metrics = self.model.evaluate_on_batch(x, y)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_predict(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        pred_y = self.model.predict(x, batch_size=ModelTest.batch_size)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_predict_with_np_array(self):
        x = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        pred_y = self.model.predict(x, batch_size=ModelTest.batch_size)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_predict_data_loader(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        generator = DataLoader(x, ModelTest.batch_size)
        pred_y = self.model.predict_generator(generator)
        self._test_predictions_for_evaluate_and_predict_generator(pred_y)

    def test_predict_generator(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        generator = (x for x, _ in generator)
        pred_y = self.model.predict_generator(generator, steps=num_steps)
        for pred in pred_y:
            self.assertEqual(type(pred), np.ndarray)
            self.assertEqual(pred.shape, (ModelTest.batch_size, 1))
        self.assertEqual(np.concatenate(pred_y).shape, (num_steps * ModelTest.batch_size, 1))

    def _test_predictions_for_evaluate_and_predict_generator(self, pred_y):
        self.assertEqual(type(pred_y), list)
        remaning_example = ModelTest.evaluate_dataset_len
        cur_batch_size = ModelTest.batch_size
        for pred in pred_y:
            self.assertEqual(type(pred), np.ndarray)
            if remaning_example < ModelTest.batch_size:
                cur_batch_size = remaning_example
                remaning_example = 0
            else:
                remaning_example -= ModelTest.batch_size
            self.assertEqual(pred.shape, (cur_batch_size, 1))
        self.assertEqual(np.concatenate(pred_y).shape, (ModelTest.evaluate_dataset_len, 1))

    def test_tensor_predict_on_batch(self):
        x = torch.rand(ModelTest.batch_size, 1)
        pred_y = self.model.predict_on_batch(x)
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_ndarray_predict_on_batch(self):
        x = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        pred_y = self.model.predict_on_batch(x)
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_cpu_cuda(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)

        with torch.cuda.device(ModelTest.cuda_device):
            self.model.cuda()
            self.model.fit_generator(train_generator, valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

        # The context manager is also used here because of this bug:
        # https://github.com/pytorch/pytorch/issues/7320
        with torch.cuda.device(ModelTest.cuda_device):
            self.model.cuda(ModelTest.cuda_device)
            self._test_device(torch.device('cuda:' + str(ModelTest.cuda_device)))
            self.model.fit_generator(train_generator, valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self.model.cpu()
            self._test_device(torch.device('cpu'))
            self.model.fit_generator(train_generator, valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self.model.to(torch.device('cuda:' + str(ModelTest.cuda_device)))
            self._test_device(torch.device('cuda:' + str(ModelTest.cuda_device)))
            self.model.fit_generator(train_generator, valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self.model.to(torch.device('cpu'))
            self._test_device(torch.device('cpu'))
            self.model.fit_generator(train_generator, valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

    def _test_device(self, device):
        for p in self.pytorch_module.parameters():
            self.assertEqual(p.device, device)

    def test_disable_batch_size_warning(self):
        import warnings
        def tuple_generator(batch_size):
            while True:
                x1 = torch.rand(batch_size, 1)
                x2 = torch.rand(batch_size, 1)
                y1 = torch.rand(batch_size, 1)
                y2 = torch.rand(batch_size, 1)
                yield (x1, x2), (y1, y2)
        class TupleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(1, 1)
                self.l2 = nn.Linear(1, 1)

            def forward(self, x):
                # pylint: disable=arguments-differ
                x1, x2 = x
                return self.l1(x1), self.l2(x2)

        def loss_function(y_pred, y_true):
            return F.mse_loss(y_pred[0], y_true[0]) + F.mse_loss(y_pred[1], y_true[1])

        pytorch_module = TupleModule()
        optimizer = torch.optim.SGD(pytorch_module.parameters(), lr=1e-3)
        model = Model(pytorch_module, optimizer, loss_function)

        train_generator = tuple_generator(ModelTest.batch_size)
        valid_generator = tuple_generator(ModelTest.batch_size)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit_generator(train_generator, valid_generator,
                                epochs=ModelTest.epochs,
                                steps_per_epoch=ModelTest.steps_per_epoch,
                                validation_steps=ModelTest.steps_per_epoch)
            num_warnings = ModelTest.steps_per_epoch * 2 * ModelTest.epochs
            self.assertEqual(len(w), num_warnings)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warning_settings['batch_size'] = 'ignore'
            model.fit_generator(train_generator, valid_generator,
                                epochs=ModelTest.epochs,
                                steps_per_epoch=ModelTest.steps_per_epoch,
                                validation_steps=ModelTest.steps_per_epoch)
            self.assertEqual(len(w), 0)


if __name__ == '__main__':
    unittest.main()
