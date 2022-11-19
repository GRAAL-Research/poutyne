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

import os
from unittest import main, TestCase
from unittest.mock import ANY, Mock, call

import torch
import torch.nn as nn
from poutyne import Model
from poutyne import BaseStrategy, DefaultStrategy
from tests.framework.tools import (
    some_data_tensor_generator,
    SomeConstantMetric,
    some_batch_metric_1,
    some_batch_metric_2,
    repeat_batch_metric,
    some_metric_1_value,
    some_metric_2_value,
    repeat_batch_metric_value,
    some_constant_metric_value,
)


class StratgyIntegrationTest(TestCase):
    epochs = 10
    steps_per_epoch = 5
    batch_size = 20

    evaluate_dataset_len = 107

    cuda_device = int(os.environ.get('CUDA_DEVICE', 0))

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

        self.mock_strategy = Mock(BaseStrategy, wraps=DefaultStrategy())
        self.model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics,
            strategy=self.mock_strategy,
        )

    def test_strategy_calls_when_training(self):
        train_generator = some_data_tensor_generator(StratgyIntegrationTest.batch_size)
        valid_generator = some_data_tensor_generator(StratgyIntegrationTest.batch_size)
        logs = self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=StratgyIntegrationTest.epochs,
            steps_per_epoch=StratgyIntegrationTest.steps_per_epoch,
            validation_steps=StratgyIntegrationTest.steps_per_epoch,
        )

        params = {
            'epochs': StratgyIntegrationTest.epochs,
            'steps': StratgyIntegrationTest.steps_per_epoch,
            'valid_steps': StratgyIntegrationTest.steps_per_epoch,
        }
        expected_calls = self._get_expected_strategy_calls_when_training(params, logs)
        actual_calls = self.mock_strategy.mock_calls
        self.assertEqual(expected_calls, actual_calls)

    def test_strategy_calls_when_training_with_no_validation(self):
        train_generator = some_data_tensor_generator(StratgyIntegrationTest.batch_size)
        logs = self.model.fit_generator(
            train_generator,
            epochs=StratgyIntegrationTest.epochs,
            steps_per_epoch=StratgyIntegrationTest.steps_per_epoch,
        )

        params = {
            'epochs': StratgyIntegrationTest.epochs,
            'steps': StratgyIntegrationTest.steps_per_epoch,
        }
        expected_calls = self._get_expected_strategy_calls_when_training(params, logs, has_valid=False)
        actual_calls = self.mock_strategy.mock_calls
        self.assertEqual(expected_calls, actual_calls)

    def _get_expected_strategy_calls_when_training(self, params, logs, has_valid=True, steps=None, valid_steps=None):
        # pylint: disable=too-many-arguments, too-many-locals
        if steps is None:
            steps = params['steps']
        if has_valid and valid_steps is None:
            valid_steps = params['valid_steps']

        train_batch_dict = dict(zip(self.batch_metrics_names, self.batch_metrics_values), time=ANY, loss=ANY)
        train_epochs_dict = dict(zip(self.epoch_metrics_names, self.epoch_metrics_values))
        log_dict = {**train_batch_dict, **train_epochs_dict}
        if has_valid:
            val_batch_metrics_names = ['val_' + metric_name for metric_name in self.batch_metrics_names]
            val_batch_dict = dict(zip(val_batch_metrics_names, self.batch_metrics_values), val_loss=ANY)
            val_epoch_metrics_names = ['val_' + metric_name for metric_name in self.epoch_metrics_names]
            val_epochs_dict = dict(zip(val_epoch_metrics_names, self.epoch_metrics_values))
            log_dict.update({**val_batch_dict, **val_epochs_dict})

        call_list = []
        call_list.append(call.set_model(self.model))
        call_list.append(call.get_batch_metric_names())
        call_list.append(call.get_epoch_metric_names())
        call_list.append(call.set_params(params))
        call_list.append(call.on_train_begin({}))
        for epoch in range(1, params["epochs"] + 1):
            call_list.append(call.on_epoch_begin(epoch, {}))

            # Train loop
            for step in range(1, steps + 1):
                call_list.append(call.train_step(ANY, callback=ANY, step=step))
            call_list.append(call.compute_loss())
            call_list.append(call.reset_loss())
            call_list.append(call.compute_batch_metrics())
            call_list.append(call.reset_batch_metrics())
            call_list.append(call.compute_epoch_metrics())
            call_list.append(call.reset_epoch_metrics())

            # Validation loop
            if has_valid:
                call_list.append(call.on_valid_begin({}))
                for step in range(valid_steps):
                    call_list.append(call.test_step(ANY))
                call_list.append(call.compute_loss())
                call_list.append(call.reset_loss())
                call_list.append(call.compute_batch_metrics())
                call_list.append(call.reset_batch_metrics())
                call_list.append(call.compute_epoch_metrics())
                call_list.append(call.reset_epoch_metrics())
                call_list.append(call.on_valid_end({'time': ANY, **val_batch_dict, **val_epochs_dict}))

            call_list.append(call.on_epoch_end(epoch, logs[epoch - 1]))

        call_list.append(call.on_train_end({}))
        return call_list


if __name__ == '__main__':
    main()
