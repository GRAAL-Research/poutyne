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

from typing import List
from unittest import TestCase
from unittest.mock import MagicMock, call

import torch
from torch import nn

from poutyne import Model
from poutyne.framework.callbacks.notification import NotificationCallback
from tests.framework.tools import (
    some_data_tensor_generator,
    some_batch_metric_1,
    some_batch_metric_2,
    repeat_batch_metric,
    repeat_batch_metric_value,
    some_metric_2_value,
    some_metric_1_value,
    SomeConstantMetric,
    some_constant_metric_value,
)


class NotificationCallbackTest(TestCase):
    epochs = 10
    steps_per_epoch = 5
    batch_size = 20
    lr = 0.01

    def setUp(self) -> None:
        super().setUp()
        self.notification_callback_mock = MagicMock()
        self.notificator_mock = MagicMock()

        self.train_generator = some_data_tensor_generator(NotificationCallbackTest.batch_size)
        self.valid_generator = some_data_tensor_generator(NotificationCallbackTest.batch_size)

        self.some_x_data = torch.rand(NotificationCallbackTest.batch_size, 1)
        self.some_y_data = torch.rand(NotificationCallbackTest.batch_size, 1)

        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=NotificationCallbackTest.lr)

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

    def test_givenANotificationCallback_whenTrainingLoop_thenSendNotification(self):
        notification_callback = NotificationCallback(notificator=self.notificator_mock)
        logs = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=NotificationCallbackTest.epochs,
            steps_per_epoch=NotificationCallbackTest.steps_per_epoch,
            validation_steps=NotificationCallbackTest.steps_per_epoch,
            callbacks=[notification_callback],
        )
        call_list = self._build_notificator_call(logs)

        self._assert_mock_calls(call_list)

    def test_givenANotificationCallbackWithExperimentName_whenTrainingLoop_thenSendNotificationWithExperimentName(self):
        a_experiment_name = "A experiment name"
        notification_callback = NotificationCallback(
            notificator=self.notificator_mock, experiment_name=a_experiment_name
        )
        logs = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=NotificationCallbackTest.epochs,
            steps_per_epoch=NotificationCallbackTest.steps_per_epoch,
            validation_steps=NotificationCallbackTest.steps_per_epoch,
            callbacks=[notification_callback],
        )

        call_list = self._build_notificator_call(logs, experiment_name=a_experiment_name)

        self._assert_mock_calls(call_list)

    def test_givenANotificationCallback_whenTestLoop_thenSendNotification(self):
        notification_callback = NotificationCallback(notificator=self.notificator_mock)
        res = self.model.evaluate(
            x=self.some_x_data, y=self.some_y_data, callbacks=[notification_callback], return_dict_format=True
        )

        call_list = self._build_notificator_call(res, mode="testing")

        self._assert_mock_calls(call_list)

    def test_givenANotificationCallbackWithExperimentName_whenTestLoop_thenSendNotificationWithExperimentName(self):
        a_experiment_name = "A experiment name"
        notification_callback = NotificationCallback(
            notificator=self.notificator_mock, experiment_name=a_experiment_name
        )
        res = self.model.evaluate(
            x=self.some_x_data, y=self.some_y_data, callbacks=[notification_callback], return_dict_format=True
        )

        call_list = self._build_notificator_call(res, mode="testing", experiment_name=a_experiment_name)

        self._assert_mock_calls(call_list)

    def _assert_mock_calls(self, call_list: List):
        method_calls = self.notificator_mock.method_calls
        self.assertEqual(len(method_calls), len(call_list))
        self.assertEqual(method_calls, call_list)

    @staticmethod
    def _build_notificator_call(logs, experiment_name=None, mode: str = "training") -> List:
        experiment_name_text = f" for {experiment_name}" if experiment_name is not None else ""
        call_list = []
        call_list.append(call.send_notification('', subject=f'Start of the {mode}{experiment_name_text}.'))

        if mode == "training":
            for batch_log in logs:
                formatted_log_data = " ".join([f"{key}: {value}\n" for key, value in batch_log.items()])
                call_list.append(
                    call.send_notification(
                        f"Here the epoch metrics: \n{formatted_log_data}",
                        subject=f"Epoch {batch_log['epoch']} is done{experiment_name_text}.",
                    )
                )
                message = ''
        elif mode == "testing":
            formatted_log_data = " ".join([f"{key}: {value}\n" for key, value in logs.items()])
            message = f"Here the test metrics: \n{formatted_log_data}"

        call_list.append(call.send_notification(message, subject=f'End of the {mode}{experiment_name_text}.'))
        return call_list
