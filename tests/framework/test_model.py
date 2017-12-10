from unittest import TestCase
from unittest.mock import MagicMock, call, ANY, DEFAULT
from utils import CopyingMock

from pitoune.framework import Model
from pitoune.framework.callbacks import Callback

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import FloatTensor
from torch.utils.data import DataLoader, TensorDataset

def some_metric_1(y, y_pred):
    return FloatTensor([1])

def some_metric_2(y, y_pred):
    return FloatTensor([2])

def some_data_tensor_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y

def some_data_variable_generator(batch_size):
    while True:
        x = Variable(torch.rand(batch_size, 1))
        y = Variable(torch.rand(batch_size, 1))
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
    n_epochs = 10
    steps_per_epoch = 5

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_module = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_module.parameters(), lr=1e-3)
        self.metrics = [some_metric_1, some_metric_2]
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function, metrics=self.metrics)
        self.mock_callback = MagicMock()
        self.mock_callback.on_train_begin = CopyingMock()
        self.mock_callback.on_train_end = CopyingMock()
        self.mock_callback.on_epoch_begin = CopyingMock()
        self.mock_callback.on_epoch_end = CopyingMock()
        self.mock_callback.on_batch_begin = CopyingMock()
        self.mock_callback.on_batch_end = CopyingMock()

    def test_fitting_tensor_generator(self):
        train_generator = some_data_tensor_generator(20)
        valid_generator = some_data_tensor_generator(20)
        logs = self.model.fit_generator(train_generator, valid_generator, n_epochs=ModelTest.n_epochs, steps_per_epoch=ModelTest.steps_per_epoch, callbacks=[self.mock_callback])
        params = {'n_epochs': ModelTest.n_epochs, 'steps_per_epoch': ModelTest.steps_per_epoch, 'metrics': ['some_metric_1', 'some_metric_2']}
        self._test_fitting(params, logs)

    def test_fitting_variable_generator(self):
        train_generator = some_data_variable_generator(20)
        valid_generator = some_data_variable_generator(20)
        logs = self.model.fit_generator(train_generator, valid_generator, n_epochs=ModelTest.n_epochs, steps_per_epoch=ModelTest.steps_per_epoch, callbacks=[self.mock_callback])
        params = {'n_epochs': ModelTest.n_epochs, 'steps_per_epoch': ModelTest.steps_per_epoch, 'metrics': ['some_metric_1', 'some_metric_2']}
        self._test_fitting(params, logs)

    def test_fitting_with_data_loader(self):
        train_real_steps_per_epoch = 30
        train_batch_size = 20
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

        logs = self.model.fit_generator(train_generator, valid_generator, n_epochs=ModelTest.n_epochs, steps_per_epoch=None, callbacks=[self.mock_callback])
        params = {'n_epochs': ModelTest.n_epochs, 'steps_per_epoch': train_real_steps_per_epoch, 'metrics': ['some_metric_1', 'some_metric_2']}
        self._test_fitting(params, logs)

    def test_fitting_with_generator_with_len(self):
        train_real_steps_per_epoch = 30
        train_generator = SomeDataGeneratorWithLen(batch_size=20, length=train_real_steps_per_epoch, num_missing_samples=7)
        valid_generator = SomeDataGeneratorWithLen(batch_size=15, length=10, num_missing_samples=3)
        logs = self.model.fit_generator(train_generator, valid_generator, n_epochs=ModelTest.n_epochs, steps_per_epoch=None, callbacks=[self.mock_callback])
        params = {'n_epochs': ModelTest.n_epochs, 'steps_per_epoch': train_real_steps_per_epoch, 'metrics': ['some_metric_1', 'some_metric_2']}
        self._test_fitting(params, logs)

    def _test_fitting(self, params, logs):
        self.assertEqual(len(logs), params['n_epochs'])
        for log in logs:
            keys = ['epoch', 'lr', 'loss', 'some_metric_1', 'some_metric_2', 'val_loss', 'val_some_metric_1', 'val_some_metric_2']
            self.assertCountEqual(log.keys(), keys)
            self.assertEqual(log['some_metric_1'], 1.)
            self.assertEqual(log['some_metric_2'], 2.)
            self.assertEqual(log['val_some_metric_1'], 1.)
            self.assertEqual(log['val_some_metric_2'], 2.)

        method_calls = self.mock_callback.method_calls
        self.assertEqual(method_calls.index(call.on_train_begin([])), 2)
        self.assertIn(call.set_model(self.model), method_calls[:2])
        self.assertIn(call.set_params(params), method_calls[:2])

        i = 3
        for epoch in range(1, params['n_epochs']+1):
            self.assertEqual(method_calls[i], call.on_epoch_begin(epoch, ANY))
            self.assertEqual(len(method_calls[i][1][1]), epoch - 1)
            i += 1
            for step in range(1, params['steps_per_epoch']+1):
                self.assertEqual(method_calls[i], call.on_batch_begin(step, ANY))
                self.assertEqual(len(method_calls[i][1][1]), epoch)
                i += 1
                self.assertEqual(method_calls[i], call.on_batch_end(step, ANY))
                self.assertEqual(len(method_calls[i][1][1]), epoch)
                i += 1

            self.assertEqual(method_calls[i], call.on_epoch_end(epoch, ANY))
            self.assertEqual(len(method_calls[i][1][1]), epoch)
            i += 1

        self.assertEqual(method_calls[i], call.on_train_end(ANY))
        self.assertEqual(len(method_calls[i][1][0]), epoch)
        i += 1

        self.assertEqual(len(method_calls), i)

if __name__ == '__main__':
    unittest.main()
