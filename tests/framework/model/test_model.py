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

# pylint: disable=too-many-locals,too-many-lines

import warnings
from collections import OrderedDict
from math import ceil
from unittest import skipIf, main
from unittest.mock import MagicMock, ANY

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import DataLoader, Dataset

from poutyne import Model, warning_settings, TensorDataset
from tests.framework.tools import (
    some_data_tensor_generator,
    SomeDataGeneratorUsingStopIteration,
    SomeDataGeneratorWithLen,
    SomeConstantEpochMetric,
    some_batch_metric_1,
    some_batch_metric_2,
    repeat_batch_metric,
    some_metric_1_value,
    some_metric_2_value,
    repeat_batch_metric_value,
    some_constant_epoch_metric_value,
    SomeEpochMetric,
)
from tests.utils import populate_packed_sequence
from .base import ModelFittingTestCase

warning_settings['concatenate_returns'] = 'ignore'


def some_ndarray_generator(batch_size):
    while True:
        x = np.random.rand(batch_size, 1).astype(np.float32)
        y = np.random.rand(batch_size, 1).astype(np.float32)
        yield x, y


class IterableMock:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iter = None
        self.calls = []

    def __iter__(self):
        self.calls.append('__iter__')
        self.iter = iter(self.iterable)
        return self

    def __next__(self):
        self.calls.append('__next__')
        return next(self.iter)

    def __len__(self):
        self.calls.append('__len__')
        return len(self.iterable)


class ModelTest(ModelFittingTestCase):
    # pylint: disable=too-many-public-methods

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.pytorch_network.parameters(), lr=1e-3)
        self.batch_metrics = [
            some_batch_metric_1,
            ('custom_name', some_batch_metric_2),
            repeat_batch_metric,
            repeat_batch_metric,
        ]
        self.batch_metrics_names = [
            'some_batch_metric_1',
            'custom_name',
            'repeat_batch_metric1',
            'repeat_batch_metric2',
        ]
        self.batch_metrics_values = [
            some_metric_1_value,
            some_metric_2_value,
            repeat_batch_metric_value,
            repeat_batch_metric_value,
        ]
        self.epoch_metrics = [SomeConstantEpochMetric()]
        self.epoch_metrics_names = ['some_constant_epoch_metric']
        self.epoch_metrics_values = [some_constant_epoch_metric_value]

        self.model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics,
        )

    def test_fitting_tensor_generator(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelTest.epochs,
            steps_per_epoch=ModelTest.steps_per_epoch,
            validation_steps=ModelTest.steps_per_epoch,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelTest.epochs,
            'steps': ModelTest.steps_per_epoch,
            'valid_steps': ModelTest.steps_per_epoch,
        }
        self._test_callbacks_train(params, logs, valid_steps=ModelTest.steps_per_epoch)

    def test_fitting_without_valid_generator(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(
            train_generator,
            None,
            epochs=ModelTest.epochs,
            steps_per_epoch=ModelTest.steps_per_epoch,
            callbacks=[self.mock_callback],
        )
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_callbacks_train(params, logs, has_valid=False)

    def test_correct_optim_calls_1_batch_per_step(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)

        mocked_optimizer = MagicMock(self.optimizer)
        mocked_optim_model = Model(
            self.pytorch_network,
            mocked_optimizer,
            self.loss_function,
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics,
        )
        mocked_optim_model.fit_generator(train_generator, None, epochs=1, steps_per_epoch=1, batches_per_step=1)

        self.assertEqual(1, mocked_optimizer.step.call_count)
        self.assertEqual(1, mocked_optimizer.zero_grad.call_count)

    def test_correct_optim_calls__valid_n_batches_per_step(self):
        n_batches = 5
        items_per_batch = int(ModelTest.batch_size / n_batches)

        x = torch.rand(n_batches, items_per_batch, 1)
        y = torch.rand(n_batches, items_per_batch, 1)

        mocked_optimizer = MagicMock(self.optimizer)
        mocked_optim_model = Model(
            self.pytorch_network,
            mocked_optimizer,
            self.loss_function,
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics,
        )
        mocked_optim_model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=n_batches)

        self.assertEqual(1, mocked_optimizer.step.call_count)
        self.assertEqual(1, mocked_optimizer.zero_grad.call_count)

    def test_fitting_generator_n_batches_per_step(self):
        total_batch_size = 6

        x = torch.rand(1, total_batch_size, 1)
        y = torch.rand(1, total_batch_size, 1)

        initial_params = self.model.get_weight_copies()

        self.model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=1)

        expected_params = self.model.get_weight_copies()

        for mini_batch_size in [1, 2, 5]:
            self.model.set_weights(initial_params)

            n_batches_per_step = int(total_batch_size / mini_batch_size)

            x.resize_((n_batches_per_step, mini_batch_size, 1))
            y.resize_((n_batches_per_step, mini_batch_size, 1))

            self.model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=n_batches_per_step)

            returned_params = self.model.get_weight_copies()

            self.assertEqual(returned_params.keys(), expected_params.keys())
            for k in expected_params.keys():
                np.testing.assert_almost_equal(returned_params[k].numpy(), expected_params[k].numpy(), decimal=4)

    def test_fitting_generator_n_batches_per_step_higher_than_num_batches(self):
        total_batch_size = 6

        x = torch.rand(1, total_batch_size, 1)
        y = torch.rand(1, total_batch_size, 1)

        initial_params = self.model.get_weight_copies()

        self.model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=1)

        expected_params = self.model.get_weight_copies()

        self.model.set_weights(initial_params)

        self.model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=2)

        returned_params = self.model.get_weight_copies()

        self.assertEqual(returned_params.keys(), expected_params.keys())
        for k in expected_params.keys():
            np.testing.assert_almost_equal(returned_params[k].numpy(), expected_params[k].numpy(), decimal=4)

    def test_fitting_generator_n_batches_per_step_uneven_batches(self):
        total_batch_size = 6

        x = torch.rand(1, total_batch_size, 1)
        y = torch.rand(1, total_batch_size, 1)

        initial_params = self.model.get_weight_copies()

        self.model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=1)

        expected_params = self.model.get_weight_copies()

        x.squeeze_(dim=0)
        y.squeeze_(dim=0)

        uneven_chunk_sizes = [4, 5]

        for chunk_size in uneven_chunk_sizes:
            self.model.set_weights(initial_params)

            splitted_x = x.split(chunk_size)
            splitted_y = y.split(chunk_size)

            n_batches_per_step = ceil(total_batch_size / chunk_size)

            self.model.fit_generator(
                list(zip(splitted_x, splitted_y)), None, epochs=1, batches_per_step=n_batches_per_step
            )

            returned_params = self.model.get_weight_copies()

            self.assertEqual(returned_params.keys(), expected_params.keys())
            for k in expected_params.keys():
                np.testing.assert_almost_equal(returned_params[k].numpy(), expected_params[k].numpy(), decimal=4)

    def test_fitting_ndarray_generator(self):
        train_generator = some_ndarray_generator(ModelTest.batch_size)
        valid_generator = some_ndarray_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelTest.epochs,
            steps_per_epoch=ModelTest.steps_per_epoch,
            validation_steps=ModelTest.steps_per_epoch,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelTest.epochs,
            'steps': ModelTest.steps_per_epoch,
            'valid_steps': ModelTest.steps_per_epoch,
        }
        self._test_callbacks_train(params, logs, valid_steps=ModelTest.steps_per_epoch)

    def test_fitting_with_data_loader(self):
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples
        train_x = torch.rand(train_size, 1)
        train_y = torch.rand(train_size, 1)
        train_dataset = TensorDataset(train_x, train_y)
        train_generator = DataLoader(train_dataset, train_batch_size)

        valid_real_steps_per_epoch = 10
        valid_batch_size = 15
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples
        valid_x = torch.rand(valid_size, 1)
        valid_y = torch.rand(valid_size, 1)
        valid_dataset = TensorDataset(valid_x, valid_y)
        valid_generator = DataLoader(valid_dataset, valid_batch_size)

        logs = self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelTest.epochs,
            steps_per_epoch=None,
            validation_steps=None,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelTest.epochs,
            'steps': train_real_steps_per_epoch,
            'valid_steps': valid_real_steps_per_epoch,
        }
        self._test_callbacks_train(params, logs)

    def test_fitting_generator_calls(self):
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples
        train_x = torch.rand(train_size, 1)
        train_y = torch.rand(train_size, 1)
        train_dataset = TensorDataset(train_x, train_y)
        train_generator = DataLoader(train_dataset, train_batch_size)

        valid_real_steps_per_epoch = 10
        valid_batch_size = 15
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples
        valid_x = torch.rand(valid_size, 1)
        valid_y = torch.rand(valid_size, 1)
        valid_dataset = TensorDataset(valid_x, valid_y)
        valid_generator = DataLoader(valid_dataset, valid_batch_size)

        mock_train_generator = IterableMock(train_generator)
        mock_valid_generator = IterableMock(valid_generator)
        self.model.fit_generator(mock_train_generator, mock_valid_generator, epochs=ModelTest.epochs)
        expected_train_calls = ['__len__'] + (
            ['__iter__'] + ['__next__'] * train_real_steps_per_epoch
        ) * ModelTest.epochs
        expected_valid_calls = ['__len__'] + (
            ['__iter__'] + ['__next__'] * valid_real_steps_per_epoch
        ) * ModelTest.epochs
        self.assertEqual(mock_train_generator.calls, expected_train_calls)
        self.assertEqual(mock_valid_generator.calls, expected_valid_calls)

    def test_fitting_generator_calls_with_longer_validation_set(self):
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples
        train_x = torch.rand(train_size, 1)
        train_y = torch.rand(train_size, 1)
        train_dataset = TensorDataset(train_x, train_y)
        train_generator = DataLoader(train_dataset, train_batch_size)

        valid_real_steps_per_epoch = 40
        valid_batch_size = 15
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples
        valid_x = torch.rand(valid_size, 1)
        valid_y = torch.rand(valid_size, 1)
        valid_dataset = TensorDataset(valid_x, valid_y)
        valid_generator = DataLoader(valid_dataset, valid_batch_size)

        mock_train_generator = IterableMock(train_generator)
        mock_valid_generator = IterableMock(valid_generator)
        self.model.fit_generator(mock_train_generator, mock_valid_generator, epochs=ModelTest.epochs)
        expected_train_calls = ['__len__'] + (
            ['__iter__'] + ['__next__'] * train_real_steps_per_epoch
        ) * ModelTest.epochs
        expected_valid_calls = ['__len__'] + (
            ['__iter__'] + ['__next__'] * valid_real_steps_per_epoch
        ) * ModelTest.epochs
        self.assertEqual(mock_train_generator.calls, expected_train_calls)
        self.assertEqual(mock_valid_generator.calls, expected_valid_calls)

    def test_fitting_with_tensor(self):
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples
        train_x = torch.rand(train_size, 1)
        train_y = torch.rand(train_size, 1)

        valid_real_steps_per_epoch = 10
        # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_batch_size = train_batch_size
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples
        valid_x = torch.rand(valid_size, 1)
        valid_y = torch.rand(valid_size, 1)

        logs = self.model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=ModelTest.epochs,
            batch_size=train_batch_size,
            steps_per_epoch=None,
            validation_steps=None,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelTest.epochs,
            'steps': train_real_steps_per_epoch,
            'valid_steps': valid_real_steps_per_epoch,
        }
        self._test_callbacks_train(params, logs)

    def test_fitting_with_np_array(self):
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples
        train_x = np.random.rand(train_size, 1).astype(np.float32)
        train_y = np.random.rand(train_size, 1).astype(np.float32)

        valid_real_steps_per_epoch = 10
        # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_batch_size = train_batch_size
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples
        valid_x = np.random.rand(valid_size, 1).astype(np.float32)
        valid_y = np.random.rand(valid_size, 1).astype(np.float32)

        logs = self.model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=ModelTest.epochs,
            batch_size=train_batch_size,
            steps_per_epoch=None,
            validation_steps=None,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelTest.epochs,
            'steps': train_real_steps_per_epoch,
            'valid_steps': valid_real_steps_per_epoch,
        }
        self._test_callbacks_train(params, logs)

    def test_fitting_with_generator_with_len(self):
        train_real_steps_per_epoch = 30
        train_generator = SomeDataGeneratorWithLen(
            batch_size=ModelTest.batch_size, length=train_real_steps_per_epoch, num_missing_samples=7
        )
        valid_real_steps_per_epoch = 10
        valid_generator = SomeDataGeneratorWithLen(
            batch_size=15, length=valid_real_steps_per_epoch, num_missing_samples=3
        )
        logs = self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelTest.epochs,
            steps_per_epoch=None,
            validation_steps=None,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelTest.epochs,
            'steps': train_real_steps_per_epoch,
            'valid_steps': valid_real_steps_per_epoch,
        }
        self._test_callbacks_train(params, logs)

    def test_fitting_with_generator_with_stop_iteration(self):
        train_real_steps_per_epoch = 30
        train_generator = SomeDataGeneratorUsingStopIteration(
            batch_size=ModelTest.batch_size, length=train_real_steps_per_epoch
        )
        valid_generator = SomeDataGeneratorUsingStopIteration(batch_size=15, length=10)
        logs = self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelTest.epochs,
            steps_per_epoch=None,
            validation_steps=None,
            callbacks=[self.mock_callback],
        )
        params = {'epochs': ModelTest.epochs, 'steps': None}
        self._test_callbacks_train(params, logs, steps=train_real_steps_per_epoch)

    def test_tensor_train_on_batch(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        loss, metrics = self.model.train_on_batch(x, y)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values)

    def test_train_on_batch_with_pred(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        loss, metrics, pred_y = self.model.train_on_batch(x, y, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values)
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_ndarray_train_on_batch(self):
        x = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        y = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        loss, metrics = self.model.train_on_batch(x, y)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values)

    def test_train_on_batch_with_return_dict(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        logs = self.model.train_on_batch(x, y, return_dict_format=True)
        self.assertEqual(set(logs.keys()), set(['loss'] + self.batch_metrics_names))

    def test_train_on_batch_with_return_dict_and_pred(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        logs, pred_y = self.model.train_on_batch(x, y, return_dict_format=True, return_pred=True)
        self.assertEqual(set(logs.keys()), set(['loss'] + self.batch_metrics_names))
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_evaluate(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        loss, metrics = self.model.evaluate(x, y, batch_size=ModelTest.batch_size)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values + self.epoch_metrics_values)

    def test_evaluate_with_pred(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        # We also test the unpacking.
        _, _, pred_y = self.model.evaluate(x, y, batch_size=ModelTest.batch_size, return_pred=True)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_pred_without_convert_to_numpy(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        # We also test the unpacking.
        _, _, pred_y = self.model.evaluate(
            x, y, batch_size=ModelTest.batch_size, return_pred=True, convert_to_numpy=False
        )
        self.assertTrue(torch.is_tensor(pred_y))
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_callback(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        # We also test the unpacking.
        _, _, pred_y = self.model.evaluate(
            x, y, batch_size=ModelTest.batch_size, return_pred=True, callbacks=[self.mock_callback]
        )
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_return_dict(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        logs = self.model.evaluate(x, y, batch_size=ModelTest.batch_size, return_dict_format=True)

        self._test_return_dict_logs(logs)

    def test_evaluate_with_return_dict_and_pred(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        logs, pred_y = self.model.evaluate(
            x, y, batch_size=ModelTest.batch_size, return_dict_format=True, return_pred=True
        )

        self._test_return_dict_logs(logs)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_np_array(self):
        x = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        y = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        loss, metrics, pred_y = self.model.evaluate(x, y, batch_size=ModelTest.batch_size, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values + self.epoch_metrics_values)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_data_loader(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        dataset = TensorDataset(x, y)
        generator = DataLoader(dataset, ModelTest.batch_size)
        loss, metrics, pred_y = self.model.evaluate_generator(generator, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values + self.epoch_metrics_values)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_generator(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        loss, metrics, pred_y = self.model.evaluate_generator(generator, steps=num_steps, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values + self.epoch_metrics_values)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_generator_with_stop_iteration(self):
        test_generator = SomeDataGeneratorUsingStopIteration(ModelTest.batch_size, 10)

        loss, _ = self.model.evaluate_generator(test_generator)

        self.assertEqual(type(loss), float)

    def test_evaluate_generator_with_callback(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        self.model.evaluate_generator(generator, steps=num_steps, callbacks=[self.mock_callback])

        params = {'steps': ModelTest.epochs}
        self._test_callbacks_test(params)

    def test_evaluate_generator_with_return_dict(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = self.model.evaluate_generator(generator, steps=num_steps, return_dict_format=True)

        self._test_return_dict_logs(logs)

    def test_evaluate_generator_with_return_dict_and_pred(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        logs, pred_y = self.model.evaluate_generator(
            generator, steps=num_steps, return_dict_format=True, return_pred=True
        )

        self._test_return_dict_logs(logs)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_generator_with_return_dict_and_ground_truth(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        logs, true_y = self.model.evaluate_generator(
            generator, steps=num_steps, return_dict_format=True, return_ground_truth=True
        )

        self._test_return_dict_logs(logs)
        self.assertEqual(type(true_y), np.ndarray)
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_generator_with_return_dict_and_pred_and_ground_truth(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        logs, pred_y, true_y = self.model.evaluate_generator(
            generator, steps=num_steps, return_dict_format=True, return_pred=True, return_ground_truth=True
        )

        self._test_return_dict_logs(logs)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(type(true_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 1))
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_generator_with_return_dict_and_pred_and_ground_truth_without_convert_to_numpy(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        logs, pred_y, true_y = self.model.evaluate_generator(
            generator,
            steps=num_steps,
            return_dict_format=True,
            return_pred=True,
            return_ground_truth=True,
            convert_to_numpy=False,
        )

        self._test_return_dict_logs(logs)
        self.assertTrue(torch.is_tensor(pred_y))
        self.assertTrue(torch.is_tensor(true_y))
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 1))
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_generator_with_ground_truth(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        loss, metrics, pred_y, true_y = self.model.evaluate_generator(
            generator, steps=num_steps, return_pred=True, return_ground_truth=True
        )
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values + self.epoch_metrics_values)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(type(true_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 1))
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_generator_with_no_concatenation(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        loss, metrics, pred_y, true_y = self.model.evaluate_generator(
            generator, steps=num_steps, return_pred=True, return_ground_truth=True, concatenate_returns=False
        )
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values + self.epoch_metrics_values)

        self.assertEqual(type(pred_y), list)
        for pred in pred_y:
            self.assertEqual(type(pred), np.ndarray)
            self.assertEqual(pred.shape, (ModelTest.batch_size, 1))
        self.assertEqual(type(true_y), list)
        for true in true_y:
            self.assertEqual(type(true), np.ndarray)
            self.assertEqual(true.shape, (ModelTest.batch_size, 1))

    def test_evaluate_with_only_one_metric(self):
        model = Model(self.pytorch_network, self.optimizer, self.loss_function, batch_metrics=self.batch_metrics[:1])
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        loss, first_metric = model.evaluate(x, y, batch_size=ModelTest.batch_size)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(first_metric), float)
        self.assertEqual(first_metric, some_metric_1_value)

    def test_metrics_integration(self):
        num_steps = 10
        model = Model(self.pytorch_network, self.optimizer, self.loss_function, batch_metrics=[F.mse_loss])
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)
        model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelTest.epochs,
            steps_per_epoch=ModelTest.steps_per_epoch,
            validation_steps=ModelTest.steps_per_epoch,
            callbacks=[self.mock_callback],
        )
        generator = some_data_tensor_generator(ModelTest.batch_size)
        loss, mse = model.evaluate_generator(generator, steps=num_steps)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(mse), float)

    def test_epoch_metrics_integration(self):
        model = Model(self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=[SomeEpochMetric()])
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = model.fit_generator(
            train_generator,
            valid_generator,
            epochs=1,
            steps_per_epoch=ModelTest.steps_per_epoch,
            validation_steps=ModelTest.steps_per_epoch,
        )
        actual_value = logs[-1]['some_epoch_metric']
        val_actual_value = logs[-1]['val_some_epoch_metric']
        expected_value = 5
        self.assertEqual(val_actual_value, expected_value)
        self.assertEqual(actual_value, expected_value)

    def test_evaluate_with_no_metric(self):
        model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        loss = model.evaluate(x, y, batch_size=ModelTest.batch_size)
        self.assertEqual(type(loss), float)

    def test_tensor_evaluate_on_batch(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        loss, metrics = self.model.evaluate_on_batch(x, y)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values)

    def test_evaluate_on_batch_with_pred(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        loss, metrics, pred_y = self.model.evaluate_on_batch(x, y, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values)
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_ndarray_evaluate_on_batch(self):
        x = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        y = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        loss, metrics = self.model.evaluate_on_batch(x, y)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values)

    def test_evaluate_on_batch_with_return_dict(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        logs = self.model.evaluate_on_batch(x, y, return_dict_format=True)
        self.assertEqual(set(logs.keys()), set(['loss'] + self.batch_metrics_names))

    def test_evaluate_on_batch_with_return_dict_and_pred(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        logs, pred_y = self.model.evaluate_on_batch(x, y, return_dict_format=True, return_pred=True)
        self.assertEqual(set(logs.keys()), set(['loss'] + self.batch_metrics_names))
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_evaluate_on_batch_with_return_dict_and_pred_without_convert_to_numpy(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        logs, pred_y = self.model.evaluate_on_batch(
            x, y, return_dict_format=True, return_pred=True, convert_to_numpy=False
        )
        self.assertEqual(set(logs.keys()), set(['loss'] + self.batch_metrics_names))
        self.assertTrue(torch.is_tensor(pred_y))
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_predict(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        pred_y = self.model.predict(x, batch_size=ModelTest.batch_size)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_predict_with_np_array(self):
        x = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        pred_y = self.model.predict(x, batch_size=ModelTest.batch_size)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_predict_without_convert_to_numpy(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        pred_y = self.model.predict(x, batch_size=ModelTest.batch_size, convert_to_numpy=False)
        self.assertTrue(torch.is_tensor(pred_y))
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_predict_data_loader(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        generator = DataLoader(x, ModelTest.batch_size)
        pred_y = self.model.predict_generator(generator)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_predict_generator(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        generator = (x for x, _ in generator)
        pred_y = self.model.predict_generator(generator, steps=num_steps)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 1))

    def test_predict_generator_with_no_concatenation(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        generator = (x for x, _ in generator)
        pred_y = self.model.predict_generator(generator, steps=num_steps, concatenate_returns=False)
        self.assertEqual(type(pred_y), list)
        for pred in pred_y:
            self.assertEqual(type(pred), np.ndarray)
            self.assertEqual(pred.shape, (ModelTest.batch_size, 1))

    def test_predict_generator_with_has_ground_truth(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        pred_y = self.model.predict_generator(generator, steps=num_steps, has_ground_truth=True)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 1))

    def test_predict_generator_with_ground_truth(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        pred_y, true_y = self.model.predict_generator(generator, steps=num_steps, return_ground_truth=True)

        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(type(true_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 1))
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size, 1))

    def test_predict_generator_with_ground_truth_tensor_without_convert_to_numpy(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        pred_y, true_y = self.model.predict_generator(
            generator, steps=num_steps, return_ground_truth=True, convert_to_numpy=False
        )

        self.assertTrue(torch.is_tensor(pred_y))
        self.assertTrue(torch.is_tensor(true_y))
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 1))
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size, 1))

    def test_predict_generator_with_ground_truth_and_no_concatenation(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        pred_y, true_y = self.model.predict_generator(
            generator, steps=num_steps, return_ground_truth=True, concatenate_returns=False
        )

        self.assertEqual(type(pred_y), list)
        for pred in pred_y:
            self.assertEqual(type(pred), np.ndarray)
            self.assertEqual(pred.shape, (ModelTest.batch_size, 1))
        self.assertEqual(type(true_y), list)
        for true in true_y:
            self.assertEqual(type(true), np.ndarray)
            self.assertEqual(true.shape, (ModelTest.batch_size, 1))

    def test_tensor_predict_on_batch(self):
        x = torch.rand(ModelTest.batch_size, 1)
        pred_y = self.model.predict_on_batch(x)
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_ndarray_predict_on_batch(self):
        x = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        pred_y = self.model.predict_on_batch(x)
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_predict_on_batch_without_convert_to_numpy(self):
        x = torch.rand(ModelTest.batch_size, 1)
        pred_y = self.model.predict_on_batch(x, convert_to_numpy=False)
        self.assertTrue(torch.is_tensor(pred_y))
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_cpu_cuda(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)

        self._capture_output()

        with torch.cuda.device(ModelTest.cuda_device):
            self.model.cuda()
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTest.epochs,
                steps_per_epoch=ModelTest.steps_per_epoch,
                validation_steps=ModelTest.steps_per_epoch,
                callbacks=[self.mock_callback],
            )

        # The context manager is also used here because of this bug:
        # https://github.com/pytorch/pytorch/issues/7320
        with torch.cuda.device(ModelTest.cuda_device):
            self.model.cuda(ModelTest.cuda_device)
            self._test_device(torch.device('cuda:' + str(ModelTest.cuda_device)))
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTest.epochs,
                steps_per_epoch=ModelTest.steps_per_epoch,
                validation_steps=ModelTest.steps_per_epoch,
                callbacks=[self.mock_callback],
            )

            self.model.cpu()
            self._test_device(torch.device('cpu'))
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTest.epochs,
                steps_per_epoch=ModelTest.steps_per_epoch,
                validation_steps=ModelTest.steps_per_epoch,
                callbacks=[self.mock_callback],
            )

            self.model.to(torch.device('cuda:' + str(ModelTest.cuda_device)))
            self._test_device(torch.device('cuda:' + str(ModelTest.cuda_device)))
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTest.epochs,
                steps_per_epoch=ModelTest.steps_per_epoch,
                validation_steps=ModelTest.steps_per_epoch,
                callbacks=[self.mock_callback],
            )

            self.model.to(torch.device('cpu'))
            self._test_device(torch.device('cpu'))
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTest.epochs,
                steps_per_epoch=ModelTest.steps_per_epoch,
                validation_steps=ModelTest.steps_per_epoch,
                callbacks=[self.mock_callback],
            )

    def test_get_batch_size(self):
        batch_size = ModelTest.batch_size
        x = np.random.rand(batch_size, 1).astype(np.float32)
        y = np.random.rand(batch_size, 1).astype(np.float32)

        batch_size2 = ModelTest.batch_size + 1
        x2 = np.random.rand(batch_size2, 1).astype(np.float32)
        y2 = np.random.rand(batch_size2, 1).astype(np.float32)

        other_batch_size = batch_size2 + 1

        inf_batch_size = self.model.get_batch_size(x, y)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = self.model.get_batch_size(x2, y2)
        self.assertEqual(inf_batch_size, batch_size2)

        inf_batch_size = self.model.get_batch_size(x, y2)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = self.model.get_batch_size(x2, y)
        self.assertEqual(inf_batch_size, batch_size2)

        inf_batch_size = self.model.get_batch_size((x, x2), y)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = self.model.get_batch_size((x2, x), y)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = self.model.get_batch_size((x, x2), (y, y2))
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = self.model.get_batch_size((x2, x), (y, y2))
        self.assertEqual(inf_batch_size, batch_size2)

        inf_batch_size = self.model.get_batch_size([x, x2], y)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = self.model.get_batch_size([x2, x], y)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = self.model.get_batch_size([x, x2], [y, y2])
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = self.model.get_batch_size([x2, x], [y, y2])
        self.assertEqual(inf_batch_size, batch_size2)

        inf_batch_size = self.model.get_batch_size({'batch_size': other_batch_size, 'x': x}, {'y': y})
        self.assertEqual(inf_batch_size, other_batch_size)

        inf_batch_size = self.model.get_batch_size({'x': x}, {'batch_size': other_batch_size, 'y': y})
        self.assertEqual(inf_batch_size, other_batch_size)

        inf_batch_size = self.model.get_batch_size({'x': x}, {'y': y})
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = self.model.get_batch_size(OrderedDict([('x1', x), ('x2', x2)]), {'y': y})
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = self.model.get_batch_size(OrderedDict([('x1', x2), ('x2', x)]), {'y': y})
        self.assertEqual(inf_batch_size, batch_size2)

        inf_batch_size = self.model.get_batch_size([1, 2, 3], {'y': y})
        self.assertEqual(inf_batch_size, batch_size)

    def test_get_batch_size_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inf_batch_size = self.model.get_batch_size([1, 2, 3], [4, 5, 6])
            self.assertEqual(inf_batch_size, 1)
            self.assertEqual(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warning_settings['batch_size'] = 'ignore'
            inf_batch_size = self.model.get_batch_size([1, 2, 3], [4, 5, 6])
            self.assertEqual(inf_batch_size, 1)
            self.assertEqual(len(w), 0)


class SomeDataset(Dataset):
    def __init__(self, length):
        super().__init__()
        self.length = length
        self.x = torch.rand(length, 1, 28, 28)  # Something like MNIST
        self.y = torch.randint(10, size=(length,))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length


class ModelDatasetMethodsTest(ModelFittingTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.train_dataset = SomeDataset(5000)
        self.valid_dataset = SomeDataset(1000)
        self.test_dataset = SomeDataset(1500)
        self.pytorch_network = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        self.batch_metrics = ['accuracy']
        self.batch_metrics_names = ['acc']
        self.batch_metrics_values = [ANY]
        self.epoch_metrics = ['f1']
        self.epoch_metrics_names = ['fscore_macro']
        self.epoch_metrics_values = [ANY]
        self.model = Model(
            self.pytorch_network,
            'sgd',
            'cross_entropy',
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics,
        )

    def test_fitting_mnist(self):
        logs = self.model.fit_dataset(
            self.train_dataset,
            self.valid_dataset,
            epochs=ModelTest.epochs,
            steps_per_epoch=ModelTest.steps_per_epoch,
            validation_steps=ModelTest.steps_per_epoch,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelTest.epochs,
            'steps': ModelTest.steps_per_epoch,
            'valid_steps': ModelTest.steps_per_epoch,
        }
        self._test_callbacks_train(params, logs, valid_steps=ModelTest.steps_per_epoch)

    def test_fitting_mnist_without_valid(self):
        logs = self.model.fit_dataset(
            self.train_dataset,
            epochs=ModelTest.epochs,
            steps_per_epoch=ModelTest.steps_per_epoch,
            validation_steps=ModelTest.steps_per_epoch,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelTest.epochs,
            'steps': ModelTest.steps_per_epoch,
            'valid_steps': ModelTest.steps_per_epoch,
        }
        self._test_callbacks_train(params, logs, has_valid=False)

    def test_evaluate_dataset(self):
        num_steps = 10
        loss, metrics, pred_y = self.model.evaluate_dataset(
            self.test_dataset, batch_size=ModelTest.batch_size, steps=num_steps, return_pred=True
        )
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values + self.epoch_metrics_values)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 10))

    def test_evaluate_dataset_with_progress_bar_coloring(self):
        num_steps = 10
        self._capture_output()

        self.model.evaluate_dataset(self.test_dataset, batch_size=ModelTest.batch_size, steps=num_steps)
        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "[94m", "\u2588"])

    def test_evaluate_dataset_with_callback(self):
        num_steps = 10
        self.model.evaluate_dataset(
            self.test_dataset, batch_size=ModelTest.batch_size, steps=num_steps, callbacks=[self.mock_callback]
        )

        params = {'steps': ModelTest.epochs}
        self._test_callbacks_test(params)

    def test_evaluate_dataset_with_return_dict(self):
        num_steps = 10
        logs = self.model.evaluate_dataset(
            self.test_dataset, batch_size=ModelTest.batch_size, steps=num_steps, return_dict_format=True
        )
        self._test_return_dict_logs(logs)

    def test_evaluate_dataset_with_return_dict_and_pred(self):
        num_steps = 10
        logs, pred_y = self.model.evaluate_dataset(
            self.test_dataset,
            batch_size=ModelTest.batch_size,
            steps=num_steps,
            return_dict_format=True,
            return_pred=True,
        )

        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 10))
        self._test_return_dict_logs(logs)

    def test_evaluate_dataset_with_return_dict_and_ground_truth(self):
        num_steps = 10
        logs, true_y = self.model.evaluate_dataset(
            self.test_dataset,
            batch_size=ModelTest.batch_size,
            steps=num_steps,
            return_dict_format=True,
            return_ground_truth=True,
        )

        self.assertEqual(type(true_y), np.ndarray)
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size,))
        self._test_return_dict_logs(logs)

    def test_evaluate_dataset_with_return_dict_and_pred_and_ground_truth(self):
        num_steps = 10
        logs, pred_y, true_y = self.model.evaluate_dataset(
            self.test_dataset,
            batch_size=ModelTest.batch_size,
            steps=num_steps,
            return_dict_format=True,
            return_pred=True,
            return_ground_truth=True,
        )

        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(type(true_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 10))
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size,))
        self._test_return_dict_logs(logs)

    def test_evaluate_dataset_with_return_dict_and_pred_and_ground_truth_without_convert_to_numpy(self):
        num_steps = 10
        logs, pred_y, true_y = self.model.evaluate_dataset(
            self.test_dataset,
            batch_size=ModelTest.batch_size,
            steps=num_steps,
            return_dict_format=True,
            return_pred=True,
            return_ground_truth=True,
            convert_to_numpy=False,
        )

        self.assertTrue(torch.is_tensor(pred_y))
        self.assertTrue(torch.is_tensor(true_y))
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 10))
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size,))
        self._test_return_dict_logs(logs)

    def test_evaluate_dataset_with_ground_truth(self):
        num_steps = 10
        loss, metrics, pred_y, true_y = self.model.evaluate_dataset(
            self.test_dataset,
            batch_size=ModelTest.batch_size,
            steps=num_steps,
            return_pred=True,
            return_ground_truth=True,
        )
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), self.batch_metrics_values + self.epoch_metrics_values)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(type(true_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 10))
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size,))

    def test_predict_dataset(self):
        class PredictDataset(Dataset):
            def __init__(self, dataset):
                super().__init__()
                self.dataset = dataset

            def __getitem__(self, index):
                return self.dataset[index][0]

            def __len__(self):
                return len(self.dataset)

        num_steps = 10
        pred_y = self.model.predict_dataset(
            PredictDataset(self.test_dataset), batch_size=ModelTest.batch_size, steps=num_steps
        )
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 10))

    def test_predict_dataset_with_has_ground_truth(self):
        num_steps = 10
        pred_y = self.model.predict_dataset(
            self.test_dataset, has_ground_truth=True, batch_size=ModelTest.batch_size, steps=num_steps
        )
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 10))

    def test_predict_dataset_with_ground_truth(self):
        num_steps = 10
        pred_y, true_y = self.model.predict_dataset(
            self.test_dataset, batch_size=ModelTest.batch_size, steps=num_steps, return_ground_truth=True
        )

        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(type(true_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 10))
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size,))

    def test_predict_dataset_with_ground_truth_without_convert_to_numpy(self):
        num_steps = 10
        pred_y, true_y = self.model.predict_dataset(
            self.test_dataset,
            batch_size=ModelTest.batch_size,
            steps=num_steps,
            return_ground_truth=True,
            convert_to_numpy=False,
        )

        self.assertTrue(torch.is_tensor(pred_y))
        self.assertTrue(torch.is_tensor(true_y))
        self.assertEqual(pred_y.shape, (num_steps * ModelTest.batch_size, 10))
        self.assertEqual(true_y.shape, (num_steps * ModelTest.batch_size,))

    def test_predict_dataset_with_ground_truth_and_no_concatenation(self):
        num_steps = 10
        pred_y, true_y = self.model.predict_dataset(
            self.test_dataset,
            batch_size=ModelTest.batch_size,
            steps=num_steps,
            return_ground_truth=True,
            concatenate_returns=False,
        )

        self.assertEqual(type(pred_y), list)
        for pred in pred_y:
            self.assertEqual(type(pred), np.ndarray)
            self.assertEqual(pred.shape, (ModelTest.batch_size, 10))
        self.assertEqual(type(true_y), list)
        for true in true_y:
            self.assertEqual(type(true), np.ndarray)
            self.assertEqual(true.shape, (ModelTest.batch_size,))

    def test_preprocess_input_with_packed_sequence_return_packed_sequence_in_tuple(self):
        x = MagicMock(spec=PackedSequence)

        actual_x = self.model.preprocess_input(x)
        self.assertTrue(isinstance(actual_x, tuple))

    def test_preprocess_input_integration_with_packed_sequence(self):
        pack_padded_sequences_vectors = populate_packed_sequence()

        actual_x = self.model.preprocess_input(pack_padded_sequences_vectors)
        self.assertTrue(isinstance(actual_x[0], PackedSequence))


if __name__ == '__main__':
    main()
