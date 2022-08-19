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
import os
from unittest import skipIf

import torch
from torch import nn

from poutyne import Model
from .base import ModelFittingTestCase
from ..tools import (
    some_data_tensor_generator,
    some_batch_metric_1,
    some_batch_metric_2,
    repeat_batch_metric,
    some_metric_1_value,
    some_metric_2_value,
    repeat_batch_metric_value,
    SomeConstantMetric,
    some_constant_metric_value,
)

TEST_MULTI_GPUS = int(os.environ.get('MULTI_GPUS', 0))


@skipIf(torch.cuda.device_count() < 2, "Need at least two gpus")
@skipIf(TEST_MULTI_GPUS != 1, "Multi-gpus test not asked for.")
class ModelTestMultiGPU(ModelFittingTestCase):
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
        self.epoch_metrics = [SomeConstantMetric()]
        self.epoch_metrics_names = ['some_constant_metric']
        self.epoch_metrics_values = [some_constant_metric_value]

        self.model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics,
        )

        self.default_main_device = ModelTestMultiGPU.cuda_device

    def _test_multiple_gpu_mode(self, devices):
        if devices == "all":
            expected = torch.cuda.device_count()
        else:
            expected = len(devices)
        self.assertEqual(len([self.model.device] + self.model.other_device), expected)

    def _test_single_gpu_mode(self):
        self.assertIsNone(self.model.other_device)
        self.assertEqual(len([self.model.device]), 1)

    def test_back_and_forth_gpu_cpu_multi_gpus(self):
        devices = "all"
        train_generator = some_data_tensor_generator(ModelTestMultiGPU.batch_size)
        valid_generator = some_data_tensor_generator(ModelTestMultiGPU.batch_size)

        with torch.cuda.device(self.default_main_device):
            self.model.to(devices)
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTestMultiGPU.epochs,
                steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
                validation_steps=ModelTestMultiGPU.steps_per_epoch,
            )
            self._test_device(torch.device('cuda:' + str(self.default_main_device)))
            self._test_multiple_gpu_mode(devices=devices)

            self.model.cpu()
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTestMultiGPU.epochs,
                steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
                validation_steps=ModelTestMultiGPU.steps_per_epoch,
            )
            self._test_device(torch.device('cpu'))
            self._test_single_gpu_mode()

            self.model.to(devices)
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTestMultiGPU.epochs,
                steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
                validation_steps=ModelTestMultiGPU.steps_per_epoch,
            )
            self._test_device(torch.device('cuda:' + str(ModelTestMultiGPU.cuda_device)))
            self._test_multiple_gpu_mode(devices=devices)

            self.model.to(torch.device('cpu'))
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTestMultiGPU.epochs,
                steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
                validation_steps=ModelTestMultiGPU.steps_per_epoch,
            )
            self._test_device(torch.device('cpu'))
            self._test_single_gpu_mode()

    def test_back_and_forth_cuda_cpu_to_multi_gpus(self):
        devices = "all"
        train_generator = some_data_tensor_generator(ModelTestMultiGPU.batch_size)
        valid_generator = some_data_tensor_generator(ModelTestMultiGPU.batch_size)

        self.model.to(devices)
        self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelTestMultiGPU.epochs,
            steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
            validation_steps=ModelTestMultiGPU.steps_per_epoch,
        )

        with torch.cuda.device(self.default_main_device):
            self.model.cuda()
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTestMultiGPU.epochs,
                steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
                validation_steps=ModelTestMultiGPU.steps_per_epoch,
            )
            self._test_device(torch.device('cuda:' + str(self.default_main_device)))
            self._test_single_gpu_mode()

            self.model.cpu()
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTestMultiGPU.epochs,
                steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
                validation_steps=ModelTestMultiGPU.steps_per_epoch,
            )
            self._test_device(torch.device('cpu'))
            self._test_single_gpu_mode()

            self.model.to(devices)
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTestMultiGPU.epochs,
                steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
                validation_steps=ModelTestMultiGPU.steps_per_epoch,
            )
            self._test_device(torch.device('cuda:' + str(ModelTestMultiGPU.cuda_device)))
            self._test_multiple_gpu_mode(devices=devices)

            self.model.cuda()
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTestMultiGPU.epochs,
                steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
                validation_steps=ModelTestMultiGPU.steps_per_epoch,
            )
            self._test_device(torch.device('cuda:' + str(self.default_main_device)))
            self._test_single_gpu_mode()

            self.model.to(torch.device('cpu'))
            self.model.fit_generator(
                train_generator,
                valid_generator,
                epochs=ModelTestMultiGPU.epochs,
                steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
                validation_steps=ModelTestMultiGPU.steps_per_epoch,
            )
            self._test_device(torch.device('cpu'))
            self._test_single_gpu_mode()

    def test_devices_settings(self):
        train_generator = some_data_tensor_generator(ModelTestMultiGPU.batch_size)
        valid_generator = some_data_tensor_generator(ModelTestMultiGPU.batch_size)

        devices = "all"
        self.model.to(devices)
        self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelTestMultiGPU.epochs,
            steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
            validation_steps=ModelTestMultiGPU.steps_per_epoch,
        )
        self._test_multiple_gpu_mode(devices=devices)

        devices = ["cuda:0", "cuda:1"]
        self.model.to(devices)
        self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelTestMultiGPU.epochs,
            steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
            validation_steps=ModelTestMultiGPU.steps_per_epoch,
        )
        self._test_multiple_gpu_mode(devices=devices)

        devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        self.model.to(devices)
        self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelTestMultiGPU.epochs,
            steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
            validation_steps=ModelTestMultiGPU.steps_per_epoch,
        )
        self._test_multiple_gpu_mode(devices=devices)

        devices = ["cuda:1"]
        self.model.to(devices)
        self.assertIsNone(self.model.other_device)
        self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelTestMultiGPU.epochs,
            steps_per_epoch=ModelTestMultiGPU.steps_per_epoch,
            validation_steps=ModelTestMultiGPU.steps_per_epoch,
        )
        self._test_device(torch.device('cuda:1'))
        self._test_single_gpu_mode()
