import os

import unittest
from unittest import TestCase, skipIf
from unittest.mock import MagicMock, call, ANY

from math import ceil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from poutyne.framework import Model, warning_settings
from poutyne.framework.metrics import EpochMetric
from poutyne.utils import TensorDataset
from poutyne.utils import _concat

# pylint: disable=too-many-lines
some_metric_1_value = 1.
some_metric_2_value = 2.


def some_batch_metric_1(y, y_pred):
    # pylint: disable=unused-argument
    return torch.FloatTensor([some_metric_1_value])


def some_batch_metric_2(y, y_pred):
    # pylint: disable=unused-argument
    return torch.FloatTensor([some_metric_2_value])


class SomeEpochMetric(EpochMetric):
    def __init__(self):
        super().__init__()
        self.__name__ = self.__class__.__name__
        self.increment = 0.0

    def __call__(self, y, y_pred):
        # pylint: disable=unused-argument
        self.increment += 1

    def get_metric(self):
        increment_value = self.increment
        self.increment = 0
        return increment_value


some_constant_epoch_metric_value = 3


class SomeConstantEpochMetric(EpochMetric):
    def __call__(self, y, y_pred):
        pass

    def get_metric(self):
        return torch.FloatTensor([some_constant_epoch_metric_value])


def some_data_tensor_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y


def some_data_tensor_generator_multi_input(batch_size):
    while True:
        x1 = torch.rand(batch_size, 1)
        x2 = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield (x1, x2), y


def some_data_tensor_generator_multi_output(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y1 = torch.rand(batch_size, 1)
        y2 = torch.rand(batch_size, 1)
        yield x, (y1, y2)


def some_data_tensor_generator_multi_io(batch_size):
    while True:
        x1 = torch.rand(batch_size, 1)
        x2 = torch.rand(batch_size, 1)
        y1 = torch.rand(batch_size, 1)
        y2 = torch.rand(batch_size, 1)
        yield (x1, x2), (y1, y2)


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


class MultiIOModel(nn.Module):
    """Model to test multiple inputs/outputs"""

    def __init__(self, num_input=2, num_output=2):
        super(MultiIOModel, self).__init__()
        inputs = []
        for _ in range(num_input):
            inputs.append(nn.Linear(1, 1))
        self.inputs = nn.ModuleList(inputs)

        outputs = []
        for _ in range(num_output):
            outputs.append(nn.Linear(num_input, 1))
        self.outputs = nn.ModuleList(outputs)

    def forward(self, *x):
        inp_to_cat = []
        for i, inp in enumerate(self.inputs):
            inp_to_cat.append(inp(x[i]))
        inp_cat = torch.cat(inp_to_cat, dim=1)

        outputs = []
        for out in self.outputs:
            outputs.append(out(inp_cat))

        outputs = outputs if len(outputs) > 1 else outputs[0]
        return outputs


class DictOutputModel(nn.Module):
    """Model to test multiple dictionnary output"""

    def __init__(self):
        super(DictOutputModel, self).__init__()
        self.input = nn.Linear(1, 1)
        self.output1 = nn.Linear(1, 1)
        self.output2 = nn.Linear(1, 1)

    def forward(self, x):
        # pylint: disable=arguments-differ
        out1 = self.output1(self.input(x))
        out2 = self.output2(self.input(x))
        return {'out1': out1, 'out2': out2}


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
        self.batch_metrics = [some_batch_metric_1, some_batch_metric_2]
        self.batch_metrics_names = ['some_batch_metric_1', 'some_batch_metric_2']
        self.batch_metrics_values = [some_metric_1_value, some_metric_2_value]
        self.epoch_metrics = [SomeConstantEpochMetric()]
        self.epoch_metrics_names = ['SomeConstantEpochMetric']
        self.epoch_metrics_values = [some_constant_epoch_metric_value]

        self.model = Model(self.pytorch_module,
                           self.optimizer,
                           self.loss_function,
                           batch_metrics=self.batch_metrics,
                           epoch_metrics=self.epoch_metrics)

        self.multi_input_model = Model(MultiIOModel(num_input=1, num_output=1),
                                       self.optimizer,
                                       self.loss_function,
                                       batch_metrics=self.batch_metrics,
                                       epoch_metrics=self.epoch_metrics)

        self.multi_output_model = Model(
            MultiIOModel(num_input=1, num_output=2),
            self.optimizer,
            lambda y_pred, y_true: self.loss_function(y_pred[0], y_true[0]) + self.loss_function(y_pred[1], y_true[1]),
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics)

        self.multi_io_model = Model(
            MultiIOModel(num_input=2, num_output=2),
            self.optimizer,
            lambda y_pred, y_true: self.loss_function(y_pred[0], y_true[0]) + self.loss_function(y_pred[1], y_true[1]),
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics)

        self.dict_output_model = Model(
            DictOutputModel(),
            self.optimizer,
            lambda y_p, y_t: self.loss_function(y_p['out1'], y_t[0]) + self.loss_function(y_p['out2'], y_t[1]),
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics)

        self.mocked_optimizer = some_mocked_optimizer()
        self.mocked_optim_model = Model(self.pytorch_module,
                                        self.mocked_optimizer,
                                        self.loss_function,
                                        batch_metrics=self.batch_metrics,
                                        epoch_metrics=self.epoch_metrics)

        self.mock_callback = MagicMock()

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
        self._test_fitting(params, logs)

    def test_fitting_tensor_generator_multi_input(self):
        train_generator = some_data_tensor_generator_multi_input(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator_multi_input(ModelTest.batch_size)
        logs = self.multi_input_model.fit_generator(train_generator,
                                                    valid_generator,
                                                    epochs=ModelTest.epochs,
                                                    steps_per_epoch=ModelTest.steps_per_epoch,
                                                    validation_steps=ModelTest.steps_per_epoch,
                                                    callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_fitting(params, logs, multi_input=True)

    def test_fitting_tensor_generator_multi_output(self):
        train_generator = some_data_tensor_generator_multi_output(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator_multi_output(ModelTest.batch_size)
        logs = self.multi_output_model.fit_generator(train_generator,
                                                     valid_generator,
                                                     epochs=ModelTest.epochs,
                                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                                     validation_steps=ModelTest.steps_per_epoch,
                                                     callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_fitting(params, logs, multi_input=False, multi_output=True)

    def test_fitting_tensor_generator_multi_io(self):
        train_generator = some_data_tensor_generator_multi_io(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator_multi_io(ModelTest.batch_size)
        logs = self.multi_io_model.fit_generator(train_generator,
                                                 valid_generator,
                                                 epochs=ModelTest.epochs,
                                                 steps_per_epoch=ModelTest.steps_per_epoch,
                                                 validation_steps=ModelTest.steps_per_epoch,
                                                 callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_fitting(params, logs, multi_input=True, multi_output=True)

    def test_fitting_without_valid_generator(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = self.model.fit_generator(train_generator,
                                        None,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=ModelTest.steps_per_epoch,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': ModelTest.steps_per_epoch}
        self._test_fitting(params, logs, has_valid=False)

    def test_correct_optim_calls_1_batch_per_step(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)

        _ = self.mocked_optim_model.fit_generator(train_generator,
                                                  None,
                                                  epochs=1,
                                                  steps_per_epoch=1,
                                                  batches_per_step=1)

        self.assertEqual(1, self.mocked_optimizer.step.call_count)
        self.assertEqual(1, self.mocked_optimizer.zero_grad.call_count)

    def test_correct_optim_calls__valid_n_batches_per_step(self):
        n_batches = 5
        items_per_batch = int(ModelTest.batch_size / n_batches)

        x = torch.rand(n_batches, items_per_batch, 1)
        y = torch.rand(n_batches, items_per_batch, 1)

        _ = self.mocked_optim_model.fit_generator(list(zip(x, y)), None, epochs=1, batches_per_step=n_batches)

        self.assertEqual(1, self.mocked_optimizer.step.call_count)
        self.assertEqual(1, self.mocked_optimizer.zero_grad.call_count)

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

        logs = self.model.fit_generator(train_generator,
                                        valid_generator,
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

        logs = self.model.fit(train_x,
                              train_y,
                              validation_data=(valid_x, valid_y),
                              epochs=ModelTest.epochs,
                              batch_size=train_batch_size,
                              steps_per_epoch=None,
                              validation_steps=None,
                              callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs)

    def test_fitting_with_tensor_multi_input(self):
        # pylint: disable=too-many-locals
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - \
                     train_final_batch_missing_samples
        train_x = (torch.rand(train_size, 1), torch.rand(train_size, 1))
        train_y = torch.rand(train_size, 1)

        valid_real_steps_per_epoch = 10
        # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_batch_size = train_batch_size
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - \
                     valid_final_batch_missing_samples
        valid_x = (torch.rand(valid_size, 1), torch.rand(valid_size, 1))
        valid_y = torch.rand(valid_size, 1)

        logs = self.multi_input_model.fit(train_x,
                                          train_y,
                                          validation_data=(valid_x, valid_y),
                                          epochs=ModelTest.epochs,
                                          batch_size=train_batch_size,
                                          steps_per_epoch=None,
                                          validation_steps=None,
                                          callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs, multi_input=True)

    def test_fitting_with_tensor_multi_output(self):
        # pylint: disable=too-many-locals
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - \
                     train_final_batch_missing_samples
        train_x = torch.rand(train_size, 1)
        train_y = (torch.rand(train_size, 1), torch.rand(train_size, 1))

        valid_real_steps_per_epoch = 10
        # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_batch_size = train_batch_size
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - \
                     valid_final_batch_missing_samples
        valid_x = torch.rand(valid_size, 1)
        valid_y = (torch.rand(valid_size, 1), torch.rand(valid_size, 1))

        logs = self.multi_output_model.fit(train_x,
                                           train_y,
                                           validation_data=(valid_x, valid_y),
                                           epochs=ModelTest.epochs,
                                           batch_size=train_batch_size,
                                           steps_per_epoch=None,
                                           validation_steps=None,
                                           callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs, multi_input=False, multi_output=True)

    def test_fitting_with_tensor_multi_output_dict(self):
        # pylint: disable=too-many-locals
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - \
                     train_final_batch_missing_samples
        train_x = torch.rand(train_size, 1)
        train_y = (torch.rand(train_size, 1), torch.rand(train_size, 1))

        valid_real_steps_per_epoch = 10
        # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_batch_size = train_batch_size
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - \
                     valid_final_batch_missing_samples
        valid_x = torch.rand(valid_size, 1)
        valid_y = (torch.rand(valid_size, 1), torch.rand(valid_size, 1))

        logs = self.dict_output_model.fit(train_x,
                                          train_y,
                                          validation_data=(valid_x, valid_y),
                                          epochs=ModelTest.epochs,
                                          batch_size=train_batch_size,
                                          steps_per_epoch=None,
                                          validation_steps=None,
                                          callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs, dict_out=True)

    def test_fitting_with_tensor_multi_io(self):
        # pylint: disable=too-many-locals
        train_real_steps_per_epoch = 30
        train_batch_size = ModelTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - \
                     train_final_batch_missing_samples
        train_x = (torch.rand(train_size, 1), torch.rand(train_size, 1))
        train_y = (torch.rand(train_size, 1), torch.rand(train_size, 1))

        valid_real_steps_per_epoch = 10
        # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_batch_size = train_batch_size
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - \
                     valid_final_batch_missing_samples
        valid_x = (torch.rand(valid_size, 1), torch.rand(valid_size, 1))
        valid_y = (torch.rand(valid_size, 1), torch.rand(valid_size, 1))

        logs = self.multi_io_model.fit(train_x,
                                       train_y,
                                       validation_data=(valid_x, valid_y),
                                       epochs=ModelTest.epochs,
                                       batch_size=train_batch_size,
                                       steps_per_epoch=None,
                                       validation_steps=None,
                                       callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': train_real_steps_per_epoch}
        self._test_fitting(params, logs, multi_input=True, multi_output=True)

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

        logs = self.model.fit(train_x,
                              train_y,
                              validation_data=(valid_x, valid_y),
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
        valid_generator = SomeDataGeneratorWithLen(batch_size=15, length=10, num_missing_samples=3)
        logs = self.model.fit_generator(train_generator,
                                        valid_generator,
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
        logs = self.model.fit_generator(train_generator,
                                        valid_generator,
                                        epochs=ModelTest.epochs,
                                        steps_per_epoch=None,
                                        validation_steps=None,
                                        callbacks=[self.mock_callback])
        params = {'epochs': ModelTest.epochs, 'steps': None}
        self._test_fitting(params, logs, steps=train_real_steps_per_epoch)

    def _test_fitting(self,
                      params,
                      logs,
                      has_valid=True,
                      steps=None,
                      multi_input=False,
                      multi_output=False,
                      dict_out=False):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-arguments
        if steps is None:
            steps = params['steps']
        self.assertEqual(len(logs), params['epochs'])
        train_batch_dict = dict(zip(self.batch_metrics_names, self.batch_metrics_values), loss=ANY, time=ANY)
        train_epochs_dict = dict(zip(self.epoch_metrics_names, self.epoch_metrics_values))
        log_dict = {**train_batch_dict, **train_epochs_dict}
        if has_valid:
            val_batch_metrics_names = ['val_' + metric_name for metric_name in self.batch_metrics_names]
            val_batch_dict = dict(zip(val_batch_metrics_names, self.batch_metrics_values), val_loss=ANY)
            val_epoch_metrics_names = ['val_' + metric_name for metric_name in self.epoch_metrics_names]
            val_epochs_dict = dict(zip(val_epoch_metrics_names, self.epoch_metrics_values))
            log_dict.update({**val_batch_dict, **val_epochs_dict})

        for epoch, log in enumerate(logs, 1):
            self.assertEqual(log, dict(log_dict, epoch=epoch))

        call_list = []
        call_list.append(call.on_train_begin({}))
        for epoch in range(1, params['epochs'] + 1):
            call_list.append(call.on_epoch_begin(epoch, {}))
            for step in range(1, steps + 1):
                call_list.append(call.on_batch_begin(step, {}))
                call_list.append(call.on_backward_end(step))
                call_list.append(call.on_batch_end(step, {'batch': step, 'size': ANY, **train_batch_dict}))
            call_list.append(call.on_epoch_end(epoch, {'epoch': epoch, **log_dict}))
        call_list.append(call.on_train_end({}))

        method_calls = self.mock_callback.method_calls
        if not dict_out:
            if multi_input and not multi_output:
                self.assertIn(call.set_model(self.multi_input_model), method_calls[:2])
            elif multi_input and multi_output:
                self.assertIn(call.set_model(self.multi_io_model), method_calls[:2])
            elif not multi_input and multi_output:
                self.assertIn(call.set_model(self.multi_output_model), method_calls[:2])
            else:
                self.assertIn(call.set_model(self.model), method_calls[:2])
        else:
            self.assertIn(call.set_model(self.dict_output_model), method_calls[:2])
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

    def test_tensor_train_on_batch_multi_input(self):
        x1 = torch.rand(ModelTest.batch_size, 1)
        x2 = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        loss, metrics = self.multi_input_model.train_on_batch((x1, x2), y)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_tensor_train_on_batch_multi_output(self):
        x = torch.rand(ModelTest.batch_size, 1)
        y1 = torch.rand(ModelTest.batch_size, 1)
        y2 = torch.rand(ModelTest.batch_size, 1)
        loss, metrics = self.multi_output_model.train_on_batch(x, (y1, y2))
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_tensor_train_on_batch_multi_io(self):
        x1 = torch.rand(ModelTest.batch_size, 1)
        x2 = torch.rand(ModelTest.batch_size, 1)
        y1 = torch.rand(ModelTest.batch_size, 1)
        y2 = torch.rand(ModelTest.batch_size, 1)
        loss, metrics = self.multi_io_model.train_on_batch((x1, x2), (y1, y2))
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_train_on_batch_with_pred_multi_input(self):
        x1 = torch.rand(ModelTest.batch_size, 1)
        x2 = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        loss, metrics, pred_y = self.multi_input_model.train_on_batch((x1, x2), y, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_ndarray_train_on_batch_multi_input(self):
        x1 = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        x2 = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        y = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        loss, metrics = self.multi_input_model.train_on_batch((x1, x2), y)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_ndarray_train_on_batch_multi_output(self):
        x = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        y1 = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        y2 = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        loss, metrics = self.multi_output_model.train_on_batch(x, (y1, y2))
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_ndarray_train_on_batch_dict_output(self):
        x = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        y1 = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        y2 = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        loss, metrics = self.dict_output_model.train_on_batch(x, (y1, y2))
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_ndarray_train_on_batch_multi_io(self):
        x1 = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        x2 = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        y1 = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        y2 = np.random.rand(ModelTest.batch_size, 1).astype(np.float32)
        loss, metrics = self.multi_io_model.train_on_batch((x1, x2), (y1, y2))
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_evaluate(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        loss, metrics = self.model.evaluate(x, y, batch_size=ModelTest.batch_size)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])

    def test_evaluate_multi_input(self):
        x = (torch.rand(ModelTest.evaluate_dataset_len, 1), torch.rand(ModelTest.evaluate_dataset_len, 1))
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        loss, metrics = self.multi_input_model.evaluate(x, y, batch_size=ModelTest.batch_size)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])

    def test_evaluate_with_pred(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        # We also test the unpacking.
        # pylint: disable=unused-variable
        loss, metrics, pred_y = self.model.evaluate(x, y, batch_size=ModelTest.batch_size, return_pred=True)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_pred_multi_input(self):
        x = (torch.rand(ModelTest.evaluate_dataset_len, 1), torch.rand(ModelTest.evaluate_dataset_len, 1))
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        # We also test the unpacking.
        # pylint: disable=unused-variable
        loss, metrics, pred_y = self.multi_input_model.evaluate(x, y, batch_size=ModelTest.batch_size, return_pred=True)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_pred_multi_output(self):
        y = (torch.rand(ModelTest.evaluate_dataset_len, 1), torch.rand(ModelTest.evaluate_dataset_len, 1))
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        # We also test the unpacking.
        # pylint: disable=unused-variable
        loss, metrics, pred_y = self.multi_output_model.evaluate(x,
                                                                 y,
                                                                 batch_size=ModelTest.batch_size,
                                                                 return_pred=True)
        for pred in pred_y:
            self.assertEqual(pred.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_pred_dict_output(self):
        y = (torch.rand(ModelTest.evaluate_dataset_len, 1), torch.rand(ModelTest.evaluate_dataset_len, 1))
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        # We also test the unpacking.
        # pylint: disable=unused-variable
        loss, metrics, pred_y = self.dict_output_model.evaluate(x, y, batch_size=ModelTest.batch_size, return_pred=True)
        for key, pred in pred_y.items():
            self.assertEqual(pred.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_pred_multi_io(self):
        x = (torch.rand(ModelTest.evaluate_dataset_len, 1), torch.rand(ModelTest.evaluate_dataset_len, 1))
        y = (torch.rand(ModelTest.evaluate_dataset_len, 1), torch.rand(ModelTest.evaluate_dataset_len, 1))
        # We also test the unpacking.
        # pylint: disable=unused-variable
        loss, metrics, pred_y = self.multi_io_model.evaluate(x, y, batch_size=ModelTest.batch_size, return_pred=True)
        for pred in pred_y:
            self.assertEqual(pred.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_np_array(self):
        x = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        y = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        loss, metrics, pred_y = self.model.evaluate(x, y, batch_size=ModelTest.batch_size, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_with_np_array_multi_input(self):
        x1 = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        x2 = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        x = (x1, x2)
        y = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        loss, metrics, pred_y = self.multi_input_model.evaluate(x, y, batch_size=ModelTest.batch_size, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_evaluate_data_loader(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        dataset = TensorDataset(x, y)
        generator = DataLoader(dataset, ModelTest.batch_size)
        loss, metrics, pred_y = self.model.evaluate_generator(generator, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])
        self._test_predictions_for_evaluate_and_predict_generator(pred_y)

    def test_evaluate_data_loader_multi_input(self):
        x1 = torch.rand(ModelTest.evaluate_dataset_len, 1)
        x2 = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        dataset = TensorDataset((x1, x2), y)
        generator = DataLoader(dataset, ModelTest.batch_size)
        loss, metrics, pred_y = self.multi_input_model.evaluate_generator(generator, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])
        self._test_predictions_for_evaluate_and_predict_generator(pred_y)

    def test_evaluate_data_loader_multi_output(self):
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y1 = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y2 = torch.rand(ModelTest.evaluate_dataset_len, 1)
        dataset = TensorDataset(x, (y1, y2))
        generator = DataLoader(dataset, ModelTest.batch_size)
        loss, metrics, pred_y = self.multi_output_model.evaluate_generator(generator, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])
        self._test_predictions_for_evaluate_and_predict_generator(pred_y, multi_output=True)

    def test_evaluate_generator(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        loss, metrics, pred_y = self.model.evaluate_generator(generator, steps=num_steps, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])
        for pred in pred_y:
            self.assertEqual(type(pred), np.ndarray)
            self.assertEqual(pred.shape, (ModelTest.batch_size, 1))
        self.assertEqual(np.concatenate(pred_y).shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_generator_with_ground_truth(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelTest.batch_size)
        loss, metrics, pred_y, true_y = self.model.evaluate_generator(generator,
                                                                      steps=num_steps,
                                                                      return_pred=True,
                                                                      return_ground_truth=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])
        for pred, true in zip(pred_y, true_y):
            self.assertEqual(type(pred), np.ndarray)
            self.assertEqual(pred.shape, (ModelTest.batch_size, 1))
            self.assertEqual(type(true), np.ndarray)
            self.assertEqual(true.shape, (ModelTest.batch_size, 1))
        self.assertEqual(np.concatenate(pred_y).shape, (num_steps * ModelTest.batch_size, 1))
        self.assertEqual(np.concatenate(true_y).shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_generator_multi_input(self):
        num_steps = 10
        generator = some_data_tensor_generator_multi_input(ModelTest.batch_size)
        loss, metrics, pred_y = self.multi_input_model.evaluate_generator(generator, steps=num_steps, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])
        self._test_size_and_type_for_generator(pred_y, (ModelTest.batch_size, 1))

        # for pred in pred_y:
        #     self.assertEqual(type(pred), np.ndarray)
        #     self.assertEqual(pred.shape, (ModelTest.batch_size, 1))
        self.assertEqual(np.concatenate(pred_y).shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_generator_multi_output(self):
        num_steps = 10
        generator = some_data_tensor_generator_multi_output(ModelTest.batch_size)
        loss, metrics, pred_y = self.multi_output_model.evaluate_generator(generator, steps=num_steps, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])
        self._test_size_and_type_for_generator(pred_y, (ModelTest.batch_size, 1))
        for pred in _concat(pred_y):
            self.assertEqual(pred.shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_generator_multi_io(self):
        num_steps = 10
        generator = some_data_tensor_generator_multi_io(ModelTest.batch_size)
        loss, metrics, pred_y = self.multi_output_model.evaluate_generator(generator, steps=num_steps, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value, some_constant_epoch_metric_value])
        self._test_size_and_type_for_generator(pred_y, (ModelTest.batch_size, 1))
        for pred in _concat(pred_y):
            self.assertEqual(pred.shape, (num_steps * ModelTest.batch_size, 1))

    def test_evaluate_with_only_one_metric(self):
        model = Model(self.pytorch_module, self.optimizer, self.loss_function, batch_metrics=self.batch_metrics[:1])
        x = torch.rand(ModelTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelTest.evaluate_dataset_len, 1)
        loss, first_metric = model.evaluate(x, y, batch_size=ModelTest.batch_size)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(first_metric), float)
        self.assertEqual(first_metric, some_metric_1_value)

    def test_metrics_integration(self):
        num_steps = 10
        model = Model(self.pytorch_module, self.optimizer, self.loss_function, batch_metrics=[F.mse_loss])
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
        model = Model(self.pytorch_module, self.optimizer, self.loss_function, epoch_metrics=[SomeEpochMetric()])
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)
        logs = model.fit_generator(train_generator,
                                   valid_generator,
                                   epochs=1,
                                   steps_per_epoch=ModelTest.steps_per_epoch,
                                   validation_steps=ModelTest.steps_per_epoch)
        actual_value = logs[-1]['SomeEpochMetric']
        val_actual_value = logs[-1]['val_SomeEpochMetric']
        expected_value = 5
        self.assertEqual(val_actual_value, expected_value)
        self.assertEqual(actual_value, expected_value)

    def test_evaluate_with_no_metric(self):
        model = Model(self.pytorch_module, self.optimizer, self.loss_function)
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
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_tensor_evaluate_on_batch_multi_input(self):
        x1 = torch.rand(ModelTest.batch_size, 1)
        x2 = torch.rand(ModelTest.batch_size, 1)
        y = torch.rand(ModelTest.batch_size, 1)
        loss, metrics = self.multi_input_model.evaluate_on_batch((x1, x2), y)
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_tensor_evaluate_on_batch_multi_output(self):
        y1 = torch.rand(ModelTest.batch_size, 1)
        y2 = torch.rand(ModelTest.batch_size, 1)
        x = torch.rand(ModelTest.batch_size, 1)
        loss, metrics = self.multi_output_model.evaluate_on_batch(x, (y1, y2))
        self.assertEqual(type(loss), float)
        self.assertEqual(type(metrics), np.ndarray)
        self.assertEqual(metrics.tolist(), [some_metric_1_value, some_metric_2_value])

    def test_tensor_evaluate_on_batch_multi_io(self):
        y = (torch.rand(ModelTest.batch_size, 1), torch.rand(ModelTest.batch_size, 1))
        x = (torch.rand(ModelTest.batch_size, 1), torch.rand(ModelTest.batch_size, 1))
        loss, metrics = self.multi_io_model.evaluate_on_batch(x, y)
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

    def test_predict_multi_input(self):
        x = (torch.rand(ModelTest.evaluate_dataset_len, 1), torch.rand(ModelTest.evaluate_dataset_len, 1))
        pred_y = self.multi_input_model.predict(x, batch_size=ModelTest.batch_size)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_predict_with_np_array(self):
        x = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        pred_y = self.model.predict(x, batch_size=ModelTest.batch_size)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_predict_with_np_array_multi_input(self):
        x1 = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        x2 = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        x = (x1, x2)
        pred_y = self.multi_input_model.predict(x, batch_size=ModelTest.batch_size)
        self.assertEqual(pred_y.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_predict_with_np_array_multi_output(self):
        x = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)

        pred_y = self.multi_output_model.predict(x, batch_size=ModelTest.batch_size)
        for pred in pred_y:
            self.assertEqual(pred.shape, (ModelTest.evaluate_dataset_len, 1))

    def test_predict_with_np_array_multi_io(self):
        x1 = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        x2 = np.random.rand(ModelTest.evaluate_dataset_len, 1).astype(np.float32)
        x = (x1, x2)
        pred_y = self.multi_io_model.predict(x, batch_size=ModelTest.batch_size)
        for pred in pred_y:
            self.assertEqual(pred.shape, (ModelTest.evaluate_dataset_len, 1))

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

    def test_predict_generator_multi_input(self):
        num_steps = 10
        generator = some_data_tensor_generator_multi_input(ModelTest.batch_size)
        generator = (x for x, _ in generator)
        pred_y = self.multi_input_model.predict_generator(generator, steps=num_steps)
        for pred in pred_y:
            self.assertEqual(type(pred), np.ndarray)
            self.assertEqual(pred.shape, (ModelTest.batch_size, 1))
        self.assertEqual(np.concatenate(pred_y).shape, (num_steps * ModelTest.batch_size, 1))

    def test_predict_generator_multi_output(self):
        num_steps = 10
        generator = some_data_tensor_generator_multi_output(ModelTest.batch_size)
        generator = (x for x, _ in generator)
        pred_y = self.multi_output_model.predict_generator(generator, steps=num_steps)

        for pred in pred_y:
            self._test_size_and_type_for_generator(pred, (ModelTest.batch_size, 1))
            # self.assertEqual(type(pred), np.ndarray)
            # self.assertEqual(pred.shape, (ModelTest.batch_size, 1))
        for pred in _concat(pred_y):
            self.assertEqual(pred.shape, (num_steps * ModelTest.batch_size, 1))

    def test_predict_generator_multi_io(self):
        num_steps = 10
        generator = some_data_tensor_generator_multi_io(ModelTest.batch_size)
        generator = (x for x, _ in generator)
        pred_y = self.multi_io_model.predict_generator(generator, steps=num_steps)

        for pred in pred_y:
            self._test_size_and_type_for_generator(pred, (ModelTest.batch_size, 1))

        for pred in _concat(pred_y):
            self.assertEqual(pred.shape, (num_steps * ModelTest.batch_size, 1))

    def _test_predictions_for_evaluate_and_predict_generator(self, pred_y, multi_output=False):
        self.assertEqual(type(pred_y), list)
        remaning_example = ModelTest.evaluate_dataset_len
        cur_batch_size = ModelTest.batch_size

        def down_the_rabbit_hole(obj, cur_batch_size):
            # pylint: disable=expression-not-assigned
            if isinstance(obj, (list, tuple)):
                [down_the_rabbit_hole(o, cur_batch_size) for o in obj]
            elif isinstance(obj, dict):
                [down_the_rabbit_hole(val, cur_batch_size) for val in obj.values()]
            else:
                self.assertEqual(type(obj), np.ndarray)
                self.assertEqual(obj.shape, (cur_batch_size, 1))

        for pred in pred_y:
            if remaning_example < ModelTest.batch_size:
                cur_batch_size = remaning_example
                remaning_example = 0
            else:
                remaning_example -= ModelTest.batch_size
            self._test_size_and_type_for_generator(pred, (cur_batch_size, 1))
        if multi_output:
            for pred in _concat(pred_y):
                self.assertEqual(pred.shape, (ModelTest.evaluate_dataset_len, 1))
        else:
            self.assertEqual(_concat(pred_y).shape, (ModelTest.evaluate_dataset_len, 1))

    def _test_size_and_type_for_generator(self, pred_y, expected_size):
        # pylint: disable=expression-not-assigned
        if isinstance(pred_y, (list, tuple)):
            [self._test_size_and_type_for_generator(o, expected_size) for o in pred_y]
        elif isinstance(pred_y, dict):
            [self._test_size_and_type_for_generator(val, expected_size) for val in pred_y.values()]
        else:
            self.assertEqual(type(pred_y), np.ndarray)
            self.assertEqual(pred_y.shape, expected_size)

    def test_tensor_predict_on_batch(self):
        x = torch.rand(ModelTest.batch_size, 1)
        pred_y = self.model.predict_on_batch(x)
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_tensor_predict_on_batch_multi_input(self):
        x1 = torch.rand(ModelTest.batch_size, 1)
        x2 = torch.rand(ModelTest.batch_size, 1)
        pred_y = self.multi_input_model.predict_on_batch((x1, x2))
        self.assertEqual(pred_y.shape, (ModelTest.batch_size, 1))

    def test_tensor_predict_on_batch_multi_output(self):
        x = torch.rand(ModelTest.batch_size, 1)
        pred_y = self.multi_output_model.predict_on_batch(x)
        self._test_size_and_type_for_generator(pred_y, (ModelTest.batch_size, 1))

    def test_tensor_predict_on_batch_multi_io(self):
        x = (torch.rand(ModelTest.batch_size, 1), torch.rand(ModelTest.batch_size, 1))
        pred_y = self.multi_io_model.predict_on_batch(x)
        self._test_size_and_type_for_generator(pred_y, (ModelTest.batch_size, 1))

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
        for p in self.pytorch_module.parameters():
            self.assertEqual(p.device, device)

    @unittest.skip("Not sure if this test is still relevant with multi IO")
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

            def forward(self, x1, x2):
                # pylint: disable=arguments-differ
                # x1, x2 = x
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
            model.fit_generator(train_generator,
                                valid_generator,
                                epochs=ModelTest.epochs,
                                steps_per_epoch=ModelTest.steps_per_epoch,
                                validation_steps=ModelTest.steps_per_epoch)
            num_warnings = ModelTest.steps_per_epoch * 2 * ModelTest.epochs
            self.assertEqual(len(w), num_warnings)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warning_settings['batch_size'] = 'ignore'
            model.fit_generator(train_generator,
                                valid_generator,
                                epochs=ModelTest.epochs,
                                steps_per_epoch=ModelTest.steps_per_epoch,
                                validation_steps=ModelTest.steps_per_epoch)
            self.assertEqual(len(w), 0)


if __name__ == '__main__':
    unittest.main()
