# pylint: disable=unused-argument
from unittest import skipIf

import torch
from torch import nn
from torch.nn import DataParallel

from poutyne.framework import Model
from poutyne.framework.metrics import EpochMetric
from tests.framework.model.base import ModelFittingTestCase

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


@skipIf(torch.cuda.device_count() < 2, "Need at least two gpus")
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

        self.default_main_device = ModelTest.cuda_device

    def _test_multiple_gpu_module(self, devices):
        self.assertTrue(isinstance(self.model.network, DataParallel))
        if devices == "all":
            expected = torch.cuda.device_count()
        else:
            expected = len(devices)
        self.assertEqual(len(self.model.network.device_ids), expected)

    def test_back_and_forth_gpu_cpu_multi_gpus(self):
        devices = "all"
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)

        with torch.cuda.device(self.default_main_device):
            self.model.to(devices)
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self._test_device(torch.device('cuda:' + str(self.default_main_device)))
            self._test_multiple_gpu_module(devices=devices)

            self.model.cpu()
            self._test_device(torch.device('cpu'))
            self.assertFalse(isinstance(self.model.network, DataParallel))
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self.model.to(devices)
            self._test_device(torch.device('cuda:' + str(ModelTest.cuda_device)))
            self._test_multiple_gpu_module(devices=devices)
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self.model.to(torch.device('cpu'))
            self._test_device(torch.device('cpu'))
            self.assertFalse(isinstance(self.model.network, DataParallel))
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

    def test_back_and_forth_gpu_cpu_with_cuda_multi_gpus(self):
        devices = "all"
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)

        self.model.to(devices)
        self.model.fit_generator(train_generator,
                                 valid_generator,
                                 epochs=ModelTest.epochs,
                                 steps_per_epoch=ModelTest.steps_per_epoch,
                                 validation_steps=ModelTest.steps_per_epoch,
                                 callbacks=[self.mock_callback])

        with torch.cuda.device(self.default_main_device):
            self.model.cuda()
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self._test_device(torch.device('cuda:' + str(self.default_main_device)))
            self._test_multiple_gpu_module(devices=devices)

            self.model.cpu()
            self._test_device(torch.device('cpu'))
            self.assertFalse(isinstance(self.model.network, DataParallel))
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self.model.to(devices)
            self.model.cuda()
            self._test_device(torch.device('cuda:' + str(ModelTest.cuda_device)))
            self._test_multiple_gpu_module(devices=devices)
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

            self.model.to(torch.device('cpu'))
            self._test_device(torch.device('cpu'))
            self.assertFalse(isinstance(self.model.network, DataParallel))
            self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelTest.epochs,
                                     steps_per_epoch=ModelTest.steps_per_epoch,
                                     validation_steps=ModelTest.steps_per_epoch,
                                     callbacks=[self.mock_callback])

    def test_devices_settings(self):
        train_generator = some_data_tensor_generator(ModelTest.batch_size)
        valid_generator = some_data_tensor_generator(ModelTest.batch_size)

        devices = "all"
        self.model.to(devices)
        self.model.fit_generator(train_generator,
                                 valid_generator,
                                 epochs=ModelTest.epochs,
                                 steps_per_epoch=ModelTest.steps_per_epoch,
                                 validation_steps=ModelTest.steps_per_epoch,
                                 callbacks=[self.mock_callback])
        self._test_multiple_gpu_module(devices=devices)

        devices = ["cuda:0", "cuda:1"]
        self.model.to(devices)
        self.model.fit_generator(train_generator,
                                 valid_generator,
                                 epochs=ModelTest.epochs,
                                 steps_per_epoch=ModelTest.steps_per_epoch,
                                 validation_steps=ModelTest.steps_per_epoch,
                                 callbacks=[self.mock_callback])
        self._test_multiple_gpu_module(devices=devices)

        devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        self.model.to(devices)
        self.model.fit_generator(train_generator,
                                 valid_generator,
                                 epochs=ModelTest.epochs,
                                 steps_per_epoch=ModelTest.steps_per_epoch,
                                 validation_steps=ModelTest.steps_per_epoch,
                                 callbacks=[self.mock_callback])
        self._test_multiple_gpu_module(devices=devices)
