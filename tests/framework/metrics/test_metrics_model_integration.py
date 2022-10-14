"""
Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.
"""

# pylint: disable=unused-argument

# Bug with PyTorch source code makes torch.tensor as not callable for pylint.
# pylint: disable=not-callable
import os
import unittest
from unittest import skipIf
from unittest.mock import ANY
import numpy as np
import torch
import torch.nn as nn
import torchmetrics

from poutyne import Model, Metric, rename_doubles, do_register_metric_class, unregister_metric_class


class MyConstTorchMetric(torchmetrics.Metric):
    def __init__(self, value=0):
        super().__init__()
        self.value = value

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        pass

    def compute(self):
        return self.value


class ConstMetric(Metric):
    def __init__(self, value=0):
        super().__init__()
        self.value = value

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        pass

    def compute(self):
        return self.value

    def reset(self):
        pass


def get_batch_metric(value):
    def some_metric_name(y_pred, y_true):
        return torch.FloatTensor([value])

    return some_metric_name


class SomeMetricName(ConstMetric):
    def compute(self):
        return torch.FloatTensor([self.value])

    def reset(self):
        pass


def get_const_batch_metric(value):
    def const_batch_metric(y_pred, y_true):
        return value

    return const_batch_metric


class MetricsModelIntegrationTest(unittest.TestCase):
    # pylint: disable=too-many-public-methods
    epochs = 2
    steps_per_epoch = 3
    batch_size = 10

    cuda_device = int(os.environ.get('CUDA_DEVICE', 0))

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Sequential(nn.Linear(1, 1), nn.Flatten(0))
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        dataset_size = MetricsModelIntegrationTest.batch_size * MetricsModelIntegrationTest.steps_per_epoch
        self.train_x = torch.rand(dataset_size, 1)
        self.train_y = torch.rand(dataset_size)
        self.valid_x = torch.rand(dataset_size, 1)
        self.valid_y = torch.rand(dataset_size)

        self.metric_names = ('a', 'b')
        self.metric_values = [1, 2]

    def test_repeated_batch_metrics_handling(self):
        expected_names = ['some_metric_name1', 'some_metric_name2']
        model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            batch_metrics=[get_batch_metric(1), get_batch_metric(2)],
        )
        self._test_history(model, expected_names, [1, 2])

    def test_repeated_epoch_metrics_handling(self):
        expected_names = ['some_metric_name1', 'some_metric_name2']
        model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            epoch_metrics=[SomeMetricName(1), SomeMetricName(2)],
        )
        self._test_history(model, expected_names, [1, 2])

    def test_repeated_batch_epoch_metrics_handling(self):
        expected_names = ['some_metric_name1', 'some_metric_name2']
        model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            batch_metrics=[get_batch_metric(1)],
            epoch_metrics=[SomeMetricName(2)],
        )
        self._test_history(model, expected_names, [1, 2])

    def test_batch_metrics_with_multiple_names_returned_by_dict(self):
        d = dict(zip(self.metric_names, self.metric_values))
        batch_metric = get_const_batch_metric(d)
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, batch_metrics=[(self.metric_names, batch_metric)]
        )
        self._test_history(model, d.keys(), d.values())

    def test_epoch_metrics_with_multiple_names_returned_by_dict(self):
        d = dict(zip(self.metric_names, self.metric_values))
        epoch_metric = ConstMetric(d)
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=[(self.metric_names, epoch_metric)]
        )
        self._test_history(model, d.keys(), d.values())

    def test_batch_metrics_with_multiple_names_returned_by_tensor(self):
        batch_metric = get_const_batch_metric(torch.tensor(self.metric_values))
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, batch_metrics=[(self.metric_names, batch_metric)]
        )
        self._test_history(model, self.metric_names, self.metric_values)

    def test_epoch_metrics_with_multiple_names_returned_by_tensor(self):
        epoch_metric = ConstMetric(torch.tensor(self.metric_values))
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=[(self.metric_names, epoch_metric)]
        )
        self._test_history(model, self.metric_names, self.metric_values)

    def test_epoch_metrics_with_name_with_multiple_names_returned_by_tensor(self):
        class MetricWithName(ConstMetric):
            __name__ = self.metric_names

        epoch_metric = MetricWithName(torch.tensor(self.metric_values))
        model = Model(self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=[epoch_metric])
        self._test_history(model, self.metric_names, self.metric_values)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_batch_metrics_with_multiple_names_returned_by_tensor_on_gpu(self):
        with torch.cuda.device(MetricsModelIntegrationTest.cuda_device):
            batch_metric = get_const_batch_metric(torch.tensor(self.metric_values).cuda())
            model = Model(
                self.pytorch_network,
                self.optimizer,
                self.loss_function,
                batch_metrics=[(self.metric_names, batch_metric)],
            )
            model.cuda()
            self._test_history(model, self.metric_names, self.metric_values)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_epoch_metrics_with_multiple_names_returned_by_tensor_on_gpu(self):
        with torch.cuda.device(MetricsModelIntegrationTest.cuda_device):
            epoch_metric = ConstMetric(torch.tensor(self.metric_values).cuda())
            model = Model(
                self.pytorch_network,
                self.optimizer,
                self.loss_function,
                epoch_metrics=[(self.metric_names, epoch_metric)],
            )
            model.cuda()
            self._test_history(model, self.metric_names, self.metric_values)

    def test_batch_metrics_with_multiple_names_returned_by_ndarray(self):
        batch_metric = get_const_batch_metric(np.array(self.metric_values))
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, batch_metrics=[(self.metric_names, batch_metric)]
        )
        self._test_history(model, self.metric_names, self.metric_values)

    def test_epoch_metrics_with_multiple_names_returned_by_ndarray(self):
        epoch_metric = ConstMetric(np.array(self.metric_values))
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=[(self.metric_names, epoch_metric)]
        )
        self._test_history(model, self.metric_names, self.metric_values)

    def test_batch_metrics_with_multiple_names_returned_by_list(self):
        batch_metric = get_const_batch_metric(list(self.metric_values))
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, batch_metrics=[(self.metric_names, batch_metric)]
        )
        self._test_history(model, self.metric_names, self.metric_values)

    def test_epoch_metrics_with_multiple_names_returned_by_list(self):
        epoch_metric = ConstMetric(list(self.metric_values))
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=[(self.metric_names, epoch_metric)]
        )
        self._test_history(model, self.metric_names, self.metric_values)

    def test_batch_metrics_with_multiple_names_returned_by_tuple(self):
        batch_metric = get_const_batch_metric(tuple(self.metric_values))
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, batch_metrics=[(self.metric_names, batch_metric)]
        )
        self._test_history(model, self.metric_names, self.metric_values)

    def test_epoch_metrics_with_multiple_names_returned_by_tuple(self):
        epoch_metric = ConstMetric(tuple(self.metric_values))
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=[(self.metric_names, epoch_metric)]
        )
        self._test_history(model, self.metric_names, self.metric_values)

    def test_decomposable_metric_as_epoch_metric(self):
        batch_metric = get_const_batch_metric(tuple(self.metric_values))
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=[(self.metric_names, batch_metric)]
        )
        self._test_history(model, self.metric_names, self.metric_values)

    def test_predefined_decomposable_metric_as_epoch_metric(self):
        model = Model(self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=['mse'])
        self._test_history(model, ['mse'], [ANY])

    def test_batch_metrics_with_str_str_tuple(self):
        model = Model(self.pytorch_network, self.optimizer, self.loss_function, batch_metrics=['mse', ('mse2', 'mse')])
        self._test_history(model, ['mse', 'mse2'], [ANY, ANY])

    def test_epoch_metrics_registered_with_unique_name(self):
        names = do_register_metric_class(SomeMetricName, names=['some1', 'some2'], unique_name='unique_some')
        model = Model(
            self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=['some2', ('something', 'some1')]
        )
        self._test_history(model, ['unique_some', 'something'], [0.0, 0.0])
        unregister_metric_class(names)

    def test_torchmetrics_handling(self):
        expected_names = ['r2_score', 'spearman_corr_coef']
        model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            batch_metrics=[torchmetrics.R2Score(), torchmetrics.SpearmanCorrCoef()],
        )
        self._test_history(model, expected_names, [ANY, ANY])

    def test_repeated_torch_metrics_handling(self):
        expected_names = ['my_const_torch_metric1', 'my_const_torch_metric2']
        model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            batch_metrics=[MyConstTorchMetric(1), MyConstTorchMetric(2)],
        )
        self._test_history(model, expected_names, [1, 2])

    def test_use_torch_metrics_as_epoch_metrics(self):
        expected_names = ['metric1', 'metric2']
        model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            epoch_metrics=[('metric1', MyConstTorchMetric(1)), ('metric2', MyConstTorchMetric(2))],
        )
        self._test_history(model, expected_names, [1, 2])

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_torch_metrics_on_gpu(self):
        with torch.cuda.device(MetricsModelIntegrationTest.cuda_device):
            expected_names = ['r2_score', 'spearman_corr_coef']
            model = Model(
                self.pytorch_network,
                self.optimizer,
                self.loss_function,
                batch_metrics=[torchmetrics.R2Score(), torchmetrics.SpearmanCorrCoef()],
            )
            model.cuda()
            self._test_history(model, expected_names, [ANY, ANY])

    def _test_history(self, model, names, values):
        history = model.fit(
            self.train_x,
            self.train_y,
            validation_data=(self.valid_x, self.valid_y),
            batch_size=MetricsModelIntegrationTest.batch_size,
            epochs=MetricsModelIntegrationTest.epochs,
        )
        for logs in history:
            for name, value in zip(names, values):
                self.assertIn(name, logs)
                self.assertEqual(value, logs[name])
                self.assertIn('val_' + name, logs)
                self.assertEqual(value, logs['val_' + name])

    def test_epoch_metrics_with_str_str_tuple(self):
        dataset_size = MetricsModelIntegrationTest.batch_size * MetricsModelIntegrationTest.steps_per_epoch
        torch.manual_seed(42)
        train_x = torch.rand(dataset_size, 1)
        train_y = torch.randint(10, (dataset_size,))
        valid_x = torch.rand(dataset_size, 1)
        valid_y = torch.randint(10, (dataset_size,))

        model = Model(nn.Linear(1, 10), 'sgd', 'cross_entropy', epoch_metrics=['f1', ('f1_2', 'f1')])
        names = ['fscore_macro', 'f1_2']
        history = model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            batch_size=MetricsModelIntegrationTest.batch_size,
            epochs=MetricsModelIntegrationTest.epochs,
        )
        for logs in history:
            for name in names:
                self.assertIn(name, logs)
                self.assertEqual(ANY, logs[name])
                self.assertIn('val_' + name, logs)
                self.assertEqual(ANY, logs['val_' + name])


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
