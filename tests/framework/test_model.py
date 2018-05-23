import os
import numpy as np

from unittest import TestCase, skipIf
from unittest.mock import MagicMock, call, ANY, DEFAULT

from pytoune.framework import Model
from pytoune.framework.callbacks import Callback

import torch
import torch.nn as nn
from torch import FloatTensor
from torch.utils.data import DataLoader, TensorDataset

some_metric_1_value = 1.
some_metric_2_value = 2.

def some_metric_1(y, y_pred):
    return FloatTensor([some_metric_1_value])

def some_metric_2(y, y_pred):
    return FloatTensor([some_metric_2_value])

def some_data_tensor_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y


class SomeDataGeneratorWithLen(object):
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
        x = self.x[self.num_generator_called*self.batch_size:(self.num_generator_called+1)*self.batch_size]
        y = self.y[self.num_generator_called*self.batch_size:(self.num_generator_called+1)*self.batch_size]
        self.num_generator_called += 1
        if self.num_generator_called == self.length:
            self.num_generator_called = 0
        return x, y



class ModelTest(TestCase):
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

        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function, metrics=self.metrics)
        self.mock_callback = MagicMock()

    def test_fitting_tensor_generator(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(train_generator, valid_generator, epochs=ModelTest.epochs, steps_per_epoch=ModelTest.steps_per_epoch, validation_steps=ModelTest.steps_per_epoch, callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_fitting(params, logs)

    def test_fitting_without_valid_generator(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(train_generator, None, epochs=ModelTest.epochs, steps_per_epoch=ModelTest.steps_per_epoch, validation_steps=ModelTest.steps_per_epoch, callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_fitting(params, logs, has_valid=False)

    def test_fitting_with_data_loader(self):
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_x = torch.rand(train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples, 1)
        train_y = torch.rand(train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples, 1)
        train_dataset = TensorDataset(train_x, train_y)
        train_generator = DataLoader(train_dataset, train_batch_size)

        valid_real_steps_per_epoch = 10
        valid_batch_size = 15
        valid_final_batch_missing_samples = 3
        valid_x = torch.rand(valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples, 1)
        valid_y = torch.rand(valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples, 1)
        valid_dataset = TensorDataset(valid_x, valid_y)
        valid_generator = DataLoader(valid_dataset, valid_batch_size)

        logs = self.model.fit_generator(train_generator, valid_generator, epochs=ModelTest.epochs, steps_per_epoch=None, validation_steps=None, callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs)

    def test_fitting_with_tensor(self):
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_x = torch.rand(train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples, 1)
        train_y = torch.rand(train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples, 1)

        valid_real_steps_per_epoch = 10
        valid_batch_size = train_batch_size # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_final_batch_missing_samples = 3
        valid_x = torch.rand(valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples, 1)
        valid_y = torch.rand(valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples, 1)

        logs = self.model.fit(train_x, train_y, validation_x=valid_x, validation_y=valid_y, epochs=ModelTest.epochs, batch_size=train_batch_size, steps_per_epoch=None, validation_steps=None, callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs)

    def test_fitting_with_generator_with_len(self):
        train_real_steps_per_epoch = 30
        train_generator = SomeDataGeneratorWithLen(batch_size=ModelTest.batch_size, length=train_real_steps_per_epoch, num_missing_samples=7)
        valid_generator = SomeDataGeneratorWithLen(batch_size=15, length=10, num_missing_samples=3)
        logs = self.model.fit_generator(train_generator, valid_generator, epochs=ModelTest.epochs, steps_per_epoch=None, validation_steps=None, callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs)

    def _test_fitting(self, params, logs, has_valid=True):
        self.assertEqual(len(logs), params['epochs'])
        train_dict = dict(zip(self.metrics_names, self.metrics_values), loss=ANY)
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
            for step in range(1, params['steps']+1):
                call_list.append(call.on_batch_begin(step, {}))
                call_list.append(call.on_backward_end(step))
                call_list.append(call.on_batch_end(step, {'batch': step, 'size': ANY, **train_dict}))
            call_list.append(call.on_epoch_end(epoch, {'epoch': epoch, **log_dict}))
        call_list.append(call.on_train_end({}))

        method_calls = self.mock_callback.method_calls
        self.assertIn(call.set_model(self.model), method_calls[:2])
        self.assertIn(call.set_params(params), method_calls[:2])

        self.assertEqual(len(method_calls), len(call_list) + 2)
        self.assertEqual(method_calls[2:], call_list)

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
        loss, metrics, pred_y  = self.model.evaluate(x, y, batch_size=ModelTest.batch_size, return_pred=True)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_np_array(self):
        x = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        y = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        loss, metrics, pred_y = self.model.evaluate(x, y, batch_size=ModelTest.batch_size, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_data_loader(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        dataset = TensorDataset(x, y)
        generator = DataLoader(dataset, ModelTest.batch_size)
        loss, metrics, pred_y  = self.model.evaluate_generator(generator, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])
        self._test_predictions_for_evaluate_and_predict_generator(pred_y)

    def test_evaluate_generator(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        loss, metrics, pred_y  = self.model.evaluate_generator(generator, steps=num_steps, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])
        for pred in pred_y:
            self.assertEqual(type(pred), np.ndarray)
            self.assertEqual(pred.shape, (ModelTest.batch_size, 1))
        self.assertEqual(np.concatenate(pred_y).shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_with_only_one_metric(self):
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function, metrics=self.metrics[:1])
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        loss, first_metric = self.model.evaluate(x, y, batch_size=ModelTest.batch_size)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(first_metric), float)
        self.assertEqual(first_metric, some_metric_1_value)

    def test_evaluate_with_no_metric(self):
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function)
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        loss = self.model.evaluate(x, y, batch_size=ModelTest.batch_size)
        self.assertEqual(type(loss), float)

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
        pred_y  = self.model.predict_generator(generator)
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

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_cpu_cuda(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)

        with torch.cuda.device(ModelTest.cuda_device):
            self.model.cuda()
            logs = self.model.fit_generator(train_generator, valid_generator, epochs=ModelTest.epochs, steps_per_epoch=ModelTest.steps_per_epoch, validation_steps=ModelTest.steps_per_epoch, callbacks=[self.mock_callback])

        # The context manager is also used here because of this bug: https://github.com/pytorch/pytorch/issues/7320
        with torch.cuda.device(ModelTest.cuda_device):
            self.model.cuda(ModelTest.cuda_device)
            logs = self.model.fit_generator(train_generator, valid_generator, epochs=ModelTest.epochs, steps_per_epoch=ModelTest.steps_per_epoch, validation_steps=ModelTest.steps_per_epoch, callbacks=[self.mock_callback])

            self.model.cpu()
            logs = self.model.fit_generator(train_generator, valid_generator, epochs=ModelTest.epochs, steps_per_epoch=ModelTest.steps_per_epoch, validation_steps=ModelTest.steps_per_epoch, callbacks=[self.mock_callback])

            self.model.to(torch.device('cuda:' + str(ModelTest.cuda_device)))
            logs = self.model.fit_generator(train_generator, valid_generator, epochs=ModelTest.epochs, steps_per_epoch=ModelTest.steps_per_epoch, validation_steps=ModelTest.steps_per_epoch, callbacks=[self.mock_callback])

            self.model.to(torch.device('cpu'))
            logs = self.model.fit_generator(train_generator, valid_generator, epochs=ModelTest.epochs, steps_per_epoch=ModelTest.steps_per_epoch, validation_steps=ModelTest.steps_per_epoch, callbacks=[self.mock_callback])
