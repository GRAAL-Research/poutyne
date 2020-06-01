# pylint: disable=unused-argument,too-many-locals
import warnings
from collections import OrderedDict
from math import ceil
from unittest import skipIf, main
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from poutyne.framework import Model, warning_settings
from poutyne.framework.metrics import EpochMetric
from poutyne.utils import TensorDataset
from .base import ModelFittingTestCase

warning_settings['concatenate_returns'] = 'ignore'

some_metric_1_value = 1.
some_metric_2_value = 2.
repeat_batch_metric_value = 3.


def some_batch_metric_1(y_pred, y_true):
    return torch.FloatTensor([some_metric_1_value])


def some_batch_metric_2(y_pred, y_true):
    return torch.FloatTensor([some_metric_2_value])


def repeat_batch_metric(y_pred, y_true):
    return torch.FloatTensor([repeat_batch_metric_value])


class SomeEpochMetric(EpochMetric):
    def __init__(self):
        super().__init__()
        self.increment = 0.0

    def forward(self, y_pred, y_true):
        self.increment += 1

    def get_metric(self):
        increment_value = self.increment
        self.increment = 0
        return increment_value

    def reset(self):
        pass


some_constant_epoch_metric_value = 3


class SomeConstantEpochMetric(EpochMetric):
    def forward(self, y_pred, y_true):
        pass

    def get_metric(self):
        return torch.FloatTensor([some_constant_epoch_metric_value])

    def reset(self):
        pass


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


def some_mocked_optimizer():
    optim = MagicMock()

    return optim


class SomeDataGeneratorUsingStopIteration:
    def __init__(self, batch_size, length):
        self.batch_size = batch_size
        self.length = length

    def __iter__(self):
        return ((np.random.rand(self.batch_size, 1).astype(np.float32), np.random.rand(self.batch_size,
                                                                                       1).astype(np.float32))
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


class ModelTest(ModelFittingTestCase):
    # pylint: disable=too-many-public-methods
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.batch_metrics = [
            some_batch_metric_1, ('custom_name', some_batch_metric_2), repeat_batch_metric, repeat_batch_metric
        ]
        self.batch_metrics_names = [
            'some_batch_metric_1', 'custom_name', 'repeat_batch_metric1', 'repeat_batch_metric2'
        ]
        self.batch_metrics_values = [
            some_metric_1_value, some_metric_2_value, repeat_batch_metric_value, repeat_batch_metric_value
        ]
        self.epoch_metrics = [SomeConstantEpochMetric()]
        self.epoch_metrics_names = ['some_constant_epoch_metric']
        self.epoch_metrics_values = [some_constant_epoch_metric_value]

        self.model = Model(self.pytorch_network,
                           self.optimizer,
                           self.loss_function,
                           batch_metrics=self.batch_metrics,
                           epoch_metrics=self.epoch_metrics)

    def test_fitting_tensor_generator(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(train_generator,
                                        valid_generator,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=ModelTest.steps_per_epoch,
                                        validation_steps=ModelTest.steps_per_epoch,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_callbacks_train(params, logs)

    def test_fitting_without_valid_generator(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(train_generator,
                                        None,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=ModelTest.steps_per_epoch,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_callbacks_train(params, logs, has_valid=False)

    def test_correct_optim_calls_1_batch_per_step(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)

        mocked_optimizer = some_mocked_optimizer()
        mocked_optim_model = Model(self.pytorch_network,
                                   mocked_optimizer,
                                   self.loss_function,
                                   batch_metrics=self.batch_metrics,
                                   epoch_metrics=self.epoch_metrics)
        mocked_optim_model.fit_generator(train_generator, None, epochs=1, steps_per_epoch=1, batches_per_step=1)

        self.assertEqual(1, mocked_optimizer.step.call_count)
        self.assertEqual(1, mocked_optimizer.zero_grad.call_count)

    def test_correct_optim_calls__valid_n_batches_per_step(self):
        n_batches = 5
        items_per_batch = int(ModelTest.batch_size / n_batches)

        x = torch.rand(n_batches, items_per_batch, 1)
        y = torch.rand(n_batches, items_per_batch, 1)

        mocked_optimizer = some_mocked_optimizer()
        mocked_optim_model = Model(self.pytorch_network,
                                   mocked_optimizer,
                                   self.loss_function,
                                   batch_metrics=self.batch_metrics,
                                   epoch_metrics=self.epoch_metrics)
        mocked_optim_model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=n_batches)

        self.assertEqual(1, mocked_optimizer.step.call_count)
        self.assertEqual(1, mocked_optimizer.zero_grad.call_count)

    def test_fitting_generator_n_batches_per_step(self):
        total_batch_size = 6

        x = torch.rand(1, total_batch_size, 1)
        y = torch.rand(1, total_batch_size, 1)

        initial_params = self.model.get_weight_copies()

        self.model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=1)

        expected_params = list(self.model.get_weight_copies().values())

        for mini_batch_size in [1, 2, 5]:
            self.model.set_weights(initial_params)

            n_batches_per_step = int(total_batch_size / mini_batch_size)

            x.resize_((n_batches_per_step, mini_batch_size, 1))
            y.resize_((n_batches_per_step, mini_batch_size, 1))

            self.model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=n_batches_per_step)

            returned_params = list(self.model.get_weight_copies().values())

            np.testing.assert_almost_equal(returned_params, expected_params, decimal=4)

    def test_fitting_generator_n_batches_per_step_higher_than_num_batches(self):
        total_batch_size = 6

        x = torch.rand(1, total_batch_size, 1)
        y = torch.rand(1, total_batch_size, 1)

        initial_params = self.model.get_weight_copies()

        self.model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=1)

        expected_params = list(self.model.get_weight_copies().values())

        self.model.set_weights(initial_params)

        self.model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=2)

        returned_params = list(self.model.get_weight_copies().values())

        np.testing.assert_almost_equal(returned_params, expected_params, decimal=4)

    def test_fitting_generator_n_batches_per_step_uneven_batches(self):
        total_batch_size = 6

        x = torch.rand(1, total_batch_size, 1)
        y = torch.rand(1, total_batch_size, 1)

        initial_params = self.model.get_weight_copies()

        self.model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=1)

        expected_params = list(self.model.get_weight_copies().values())

        x.squeeze_(dim=0)
        y.squeeze_(dim=0)

        uneven_chunk_sizes = [4, 5]

        for chunk_size in uneven_chunk_sizes:
            self.model.set_weights(initial_params)

            splitted_x = x.split(chunk_size)
            splitted_y = y.split(chunk_size)

            n_batches_per_step = ceil(total_batch_size / chunk_size)

            self.model.fit_generator(list(zip(splitted_x, splitted_y)),
                                     None,
                                     epochs=1,
                                     batches_per_step=n_batches_per_step)

            returned_params = list(self.model.get_weight_copies().values())

            np.testing.assert_almost_equal(returned_params, expected_params, decimal=4)

    def test_fitting_ndarray_generator(self):
        train_generator = some_ndarray_generator(ModelTest.batch_size)
        valid_generator = some_ndarray_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(train_generator,
                                        valid_generator,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=ModelTest.steps_per_epoch,
                                        validation_steps=ModelTest.steps_per_epoch,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_callbacks_train(params, logs)

    def test_fitting_with_data_loader(self):
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

        logs = self.model.fit_generator(train_generator,
                                        valid_generator,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=None,
                                        validation_steps=None,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_callbacks_train(params, logs)

    def test_fitting_generator_calls(self):
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

        mock_train_generator = IterableMock(train_generator)
        mock_valid_generator = IterableMock(valid_generator)
        self.model.fit_generator(mock_train_generator, mock_valid_generator, epochs=ModelTest.epochs)
        expected_train_calls = ['__len__'] + \
            (['__iter__'] + ['__next__'] * train_real_steps_per_epoch) * ModelTest.epochs
        expected_valid_calls = ['__len__'] + \
            (['__iter__'] + ['__next__'] * valid_real_steps_per_epoch) * ModelTest.epochs
        self.assertEqual(mock_train_generator.calls, expected_train_calls)
        self.assertEqual(mock_valid_generator.calls, expected_valid_calls)

    def test_fitting_with_tensor(self):
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

        logs = self.model.fit(train_x,
                              train_y,
                              validation_data=(valid_x, valid_y),
                              epochs=ModelTest.epochs,
                              batch_size=train_batch_size,
                              steps_per_epoch=None,
                              validation_steps=None,
                              callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_callbacks_train(params, logs)

    def test_fitting_with_np_array(self):
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

        logs = self.model.fit(train_x,
                              train_y,
                              validation_data=(valid_x, valid_y),
                              epochs=ModelTest.epochs,
                              batch_size=train_batch_size,
                              steps_per_epoch=None,
                              validation_steps=None,
                              callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_callbacks_train(params, logs)

    def test_fitting_with_generator_with_len(self):
        train_real_steps_per_epoch = 30
        train_generator = SomeDataGeneratorWithLen(batch_size=ModelTest.batch_size,
                                                   length=train_real_steps_per_epoch,
                                                   num_missing_samples=7)
        valid_generator = SomeDataGeneratorWithLen(batch_size=15, length=10, num_missing_samples=3)
        logs = self.model.fit_generator(train_generator,
                                        valid_generator,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=None,
                                        validation_steps=None,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_callbacks_train(params, logs)

    def test_fitting_with_generator_with_stop_iteration(self):
        train_real_steps_per_epoch = 30
        train_generator = SomeDataGeneratorUsingStopIteration(batch_size=ModelTest.batch_size,
                                                              length=train_real_steps_per_epoch)
        valid_generator = SomeDataGeneratorUsingStopIteration(batch_size=15, length=10)
        logs = self.model.fit_generator(train_generator,
                                        valid_generator,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=None,
                                        validation_steps=None,
                                        callbacks=[self.mock_callback])
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

    def test_evaluate_with_callback(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        # We also test the unpacking.
        _, _, pred_y = self.model.evaluate(x,
                                           y,
                                           batch_size=ModelTest.batch_size,
                                           return_pred=True,
                                           callbacks=[self.mock_callback])
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

    def test_evaluate_generator_with_callback(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        result_log = self.model.evaluate_generator(generator,
                                                   steps=num_steps,
                                                   return_pred=True,
                                                   callbacks=[self.mock_callback])

        params = {'batch': ModelTest.epochs}
        self._test_callbacks_test(params, result_log)

    def test_evaluate_generator_with_ground_truth(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        loss, metrics, pred_y, true_y = self.model.evaluate_generator(generator,
                                                                      steps=num_steps,
                                                                      return_pred=True,
                                                                      return_ground_truth=True)
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
        loss, metrics, pred_y, true_y = self.model.evaluate_generator(generator,
                                                                      steps=num_steps,
                                                                      return_pred=True,
                                                                      return_ground_truth=True,
                                                                      concatenate_returns=False)
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
        model.fit_generator(train_generator,
                            valid_generator,
                            epochs=ModelTest.epochs,
                            steps_per_epoch=ModelTest.steps_per_epoch,
                            validation_steps=ModelTest.steps_per_epoch,
                            callbacks=[self.mock_callback])
        generator = some_data_tensor_generator(ModelTest.batch_size)
        loss, mse = model.evaluate_generator(generator, steps=num_steps)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(mse), float)

    def test_epoch_metrics_integration(self):
        model = Model(self.pytorch_network, self.optimizer, self.loss_function, epoch_metrics=[SomeEpochMetric()])
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = model.fit_generator(train_generator,
                                   valid_generator,
                                   epochs=1,
                                   steps_per_epoch=ModelTest.steps_per_epoch,
                                   validation_steps=ModelTest.steps_per_epoch)
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

    def test_predict(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        pred_y = self.model.predict(x, batch_size=ModelTest.batch_size)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_predict_with_np_array(self):
        x = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        pred_y = self.model.predict(x, batch_size=ModelTest.batch_size)
        self.assertEqual(type(pred_y), np.ndarray)
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
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

        # The context manager is also used here because of this bug:
        # https://github.com/pytorch/pytorch/issues/7320
        with torch.cuda.device(ModelTest.cuda_device):
            self.model.cuda(ModelTest.cuda_device)
            self._test_device(torch.device('cuda:' + str(ModelTest.cuda_device)))
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self.model.cpu()
            self._test_device(torch.device('cpu'))
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self.model.to(torch.device('cuda:' + str(ModelTest.cuda_device)))
            self._test_device(torch.device('cuda:' + str(ModelTest.cuda_device)))
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self.model.to(torch.device('cpu'))
            self._test_device(torch.device('cpu'))
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

    def _test_device(self, device):
        for p in self.pytorch_network.parameters():
            self.assertEqual(p.device, device)

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


if __name__ == '__main__':
    main()
