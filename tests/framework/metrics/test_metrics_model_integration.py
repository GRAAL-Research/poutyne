# pylint: disable=unused-argument

# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable
import os
import unittest
from unittest import skipIf
import numpy as np
import torch
import torch.nn as nn

from poutyne.framework import Model
from poutyne.framework.metrics import EpochMetric
from poutyne.framework.metrics.utils import rename_doubles


class ConstEpochMetric(EpochMetric):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, y_pred, y_true):
        pass

    def get_metric(self):
        return self.value

    def reset(self):
        pass


def get_batch_metric(value):
    def some_metric_name(y_pred, y_true):
        return torch.FloatTensor([value])

    return some_metric_name


class SomeMetricName(ConstEpochMetric):
    def get_metric(self):
        return torch.FloatTensor([self.value])

    def reset(self):
        pass


def get_const_batch_metric(value):
    def const_batch_metric(y_pred, y_true):
        return value

    return const_batch_metric


class MetricsModelIntegrationTest(unittest.TestCase):
    epochs = 2
    steps_per_epoch = 3
    batch_size = 10

    cuda_device = int(os.environ.get('CUDA_DEVICE', 0))

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        dataset_size = MetricsModelIntegrationTest.batch_size * MetricsModelIntegrationTest.steps_per_epoch
        self.train_x = torch.rand(dataset_size, 1)
        self.train_y = torch.rand(dataset_size, 1)
        self.valid_x = torch.rand(dataset_size, 1)
        self.valid_y = torch.rand(dataset_size, 1)

        self.metric_names = ('a', 'b')
        self.metric_values = [1, 2]

    def test_repeated_batch_metrics_handling(self):
        expected_names = ['some_metric_name1', 'some_metric_name2']
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      batch_metrics=[get_batch_metric(1), get_batch_metric(2)])
        self._test_history(model, expected_names, [1, 2])

    def test_repeated_epoch_metrics_handling(self):
        expected_names = ['some_metric_name1', 'some_metric_name2']
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      epoch_metrics=[SomeMetricName(1), SomeMetricName(2)])
        self._test_history(model, expected_names, [1, 2])

    def test_repeated_batch_epoch_metrics_handling(self):
        expected_names = ['some_metric_name1', 'some_metric_name2']
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      batch_metrics=[get_batch_metric(1)],
                      epoch_metrics=[SomeMetricName(2)])
        self._test_history(model, expected_names, [1, 2])

    def test_batch_metrics_with_multiple_names_returned_by_dict(self):
        d = dict(zip(self.metric_names, self.metric_values))
        batch_metric = get_const_batch_metric(d)
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      batch_metrics=[(self.metric_names, batch_metric)])
        self._test_history(model, d.keys(), d.values())

    def test_epoch_metrics_with_multiple_names_returned_by_dict(self):
        d = dict(zip(self.metric_names, self.metric_values))
        epoch_metric = ConstEpochMetric(d)
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      epoch_metrics=[(self.metric_names, epoch_metric)])
        self._test_history(model, d.keys(), d.values())

    def test_batch_metrics_with_multiple_names_returned_by_tensor(self):
        batch_metric = get_const_batch_metric(torch.tensor(self.metric_values))
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      batch_metrics=[(self.metric_names, batch_metric)])
        self._test_history(model, self.metric_names, self.metric_values)

    def test_epoch_metrics_with_multiple_names_returned_by_tensor(self):
        epoch_metric = ConstEpochMetric(torch.tensor(self.metric_values))
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      epoch_metrics=[(self.metric_names, epoch_metric)])
        self._test_history(model, self.metric_names, self.metric_values)

    def test_epoch_metrics_with_name_with_multiple_names_returned_by_tensor(self):
        class EpochMetricWithName(ConstEpochMetric):
            __name__ = self.metric_names

        epoch_metric = EpochMetricWithName(torch.tensor(self.metric_values))
        model = Model(self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=[epoch_metric])
        self._test_history(model, self.metric_names, self.metric_values)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_batch_metrics_with_multiple_names_returned_by_tensor_on_gpu(self):
        with torch.cuda.device(MetricsModelIntegrationTest.cuda_device):
            batch_metric = get_const_batch_metric(torch.tensor(self.metric_values).cuda())
            model = Model(self.pytorch_network,
                          self.optimizer,
                          self.loss_function,
                          batch_metrics=[(self.metric_names, batch_metric)])
            model.cuda()
            self._test_history(model, self.metric_names, self.metric_values)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_epoch_metrics_with_multiple_names_returned_by_tensor_on_gpu(self):
        with torch.cuda.device(MetricsModelIntegrationTest.cuda_device):
            epoch_metric = ConstEpochMetric(torch.tensor(self.metric_values).cuda())
            model = Model(self.pytorch_network,
                          self.optimizer,
                          self.loss_function,
                          epoch_metrics=[(self.metric_names, epoch_metric)])
            model.cuda()
            self._test_history(model, self.metric_names, self.metric_values)

    def test_batch_metrics_with_multiple_names_returned_by_ndarray(self):
        batch_metric = get_const_batch_metric(np.array(self.metric_values))
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      batch_metrics=[(self.metric_names, batch_metric)])
        self._test_history(model, self.metric_names, self.metric_values)

    def test_epoch_metrics_with_multiple_names_returned_by_ndarray(self):
        epoch_metric = ConstEpochMetric(np.array(self.metric_values))
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      epoch_metrics=[(self.metric_names, epoch_metric)])
        self._test_history(model, self.metric_names, self.metric_values)

    def test_batch_metrics_with_multiple_names_returned_by_list(self):
        batch_metric = get_const_batch_metric(list(self.metric_values))
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      batch_metrics=[(self.metric_names, batch_metric)])
        self._test_history(model, self.metric_names, self.metric_values)

    def test_epoch_metrics_with_multiple_names_returned_by_list(self):
        epoch_metric = ConstEpochMetric(list(self.metric_values))
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      epoch_metrics=[(self.metric_names, epoch_metric)])
        self._test_history(model, self.metric_names, self.metric_values)

    def test_batch_metrics_with_multiple_names_returned_by_tuple(self):
        batch_metric = get_const_batch_metric(tuple(self.metric_values))
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      batch_metrics=[(self.metric_names, batch_metric)])
        self._test_history(model, self.metric_names, self.metric_values)

    def test_epoch_metrics_with_multiple_names_returned_by_tuple(self):
        epoch_metric = ConstEpochMetric(tuple(self.metric_values))
        model = Model(self.pytorch_network,
                      self.optimizer,
                      self.loss_function,
                      epoch_metrics=[(self.metric_names, epoch_metric)])
        self._test_history(model, self.metric_names, self.metric_values)

    def _test_history(self, model, names, values):
        history = model.fit(self.train_x,
                            self.train_y,
                            validation_data=(self.valid_x, self.valid_y),
                            batch_size=MetricsModelIntegrationTest.batch_size,
                            epochs=MetricsModelIntegrationTest.epochs)
        for logs in history:
            for name, value in zip(names, values):
                self.assertIn(name, logs)
                self.assertEqual(value, logs[name])
                self.assertIn('val_' + name, logs)
                self.assertEqual(value, logs['val_' + name])


class MetricsRenamingTest(unittest.TestCase):
    def test_batch_metrics(self):
        actual = rename_doubles(['a', 'a'], [])
        expected = ['a1', 'a2'], []
        self.assertEqual(expected, actual)

        actual = rename_doubles(['a', 'b', 'a', 'a', 'c', 'd'], [])
        expected = ['a1', 'b', 'a2', 'a3', 'c', 'd'], []
        self.assertEqual(expected, actual)

    def test_epoch_metrics(self):
        actual = rename_doubles([], ['a', 'a'])
        expected = [], ['a1', 'a2']
        self.assertEqual(expected, actual)

        actual = rename_doubles([], ['a', 'b', 'a', 'a', 'c', 'd'])
        expected = [], ['a1', 'b', 'a2', 'a3', 'c', 'd']
        self.assertEqual(expected, actual)

    def test_batch_epoch_metrics(self):
        actual = rename_doubles(['a', 'b', 'c'], ['d', 'a', 'e', 'a'])
        expected = ['a1', 'b', 'c'], ['d', 'a2', 'e', 'a3']
        self.assertEqual(expected, actual)

        actual = rename_doubles(['a', 'b', 'c', 'b'], ['d', 'a', 'e', 'a', 'e'])
        expected = ['a1', 'b1', 'c', 'b2'], ['d', 'a2', 'e1', 'a3', 'e2']
        self.assertEqual(expected, actual)

        actual = rename_doubles(['a', 'b', 'c'], ['d', 'a', 'e', 'a', 'e'])
        expected = ['a1', 'b', 'c'], ['d', 'a2', 'e1', 'a3', 'e2']
        self.assertEqual(expected, actual)

    def test_nested_batch_metrics(self):
        actual = rename_doubles([['a', 'a']], [])
        expected = [['a1', 'a2']], []
        self.assertEqual(expected, actual)

        actual = rename_doubles([['a', 'b'], 'b'], [])
        expected = [['a', 'b1'], 'b2'], []
        self.assertEqual(expected, actual)

    def test_nested_epoch_metrics(self):
        actual = rename_doubles([], [['a', 'a']])
        expected = [], [['a1', 'a2']]
        self.assertEqual(expected, actual)

        actual = rename_doubles([], [['a', 'b'], 'b'])
        expected = [], [['a', 'b1'], 'b2']
        self.assertEqual(expected, actual)

    def test_nested_batch_epoch_metrics(self):
        actual = rename_doubles([['a']], [['a']])
        expected = [['a1']], [['a2']]
        self.assertEqual(expected, actual)

        actual = rename_doubles([['a', 'b']], [['c', 'a']])
        expected = [['a1', 'b']], [['c', 'a2']]
        self.assertEqual(expected, actual)

        actual = rename_doubles(['a', 'b', ['b', 'a'], 'c', 'a', 'd'], ['e', ['c', 'a'], 'f', 'a'])
        expected = ['a1', 'b1', ['b2', 'a2'], 'c1', 'a3', 'd'], ['e', ['c2', 'a4'], 'f', 'a5']
        self.assertEqual(expected, actual)
