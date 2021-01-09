from typing import Dict
from unittest import TestCase
from unittest.mock import MagicMock, ANY, call

import torch
from torch import nn

from poutyne import Model
from poutyne.framework.callbacks.notification import NotificationCallback
from tests.framework.tools import some_data_tensor_generator, some_batch_metric_1, some_batch_metric_2, \
    repeat_batch_metric, repeat_batch_metric_value, some_metric_2_value, some_metric_1_value, SomeConstantEpochMetric, \
    some_constant_epoch_metric_value


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

        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=NotificationCallbackTest.lr)

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

    def test_givenANotificationCallback_whenTrainingLoop_thenCallbackIsCallProperly(self):
        logs = self.model.fit_generator(self.train_generator,
                                        self.valid_generator,
                                        epochs=NotificationCallbackTest.epochs,
                                        steps_per_epoch=NotificationCallbackTest.steps_per_epoch,
                                        validation_steps=NotificationCallbackTest.steps_per_epoch,
                                        callbacks=[self.notification_callback_mock])

        params = {'epochs': NotificationCallbackTest.epochs, 'steps': NotificationCallbackTest.steps_per_epoch}
        self._test_callbacks_call(params, logs)

    def _test_callbacks_call(self, params: Dict, logs: Dict):
        # pylint: disable=too-many-locals
        steps = params['steps']

        self.assertEqual(len(logs), params['epochs'])
        train_batch_dict = dict(zip(self.batch_metrics_names, self.batch_metrics_values), loss=ANY, time=ANY)
        train_epochs_dict = dict(zip(self.epoch_metrics_names, self.epoch_metrics_values))
        log_dict = {**train_batch_dict, **train_epochs_dict}

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
                call_list.append(call.on_train_batch_begin(step, {}))
                call_list.append(call.on_backward_end(step))
                call_list.append(call.on_train_batch_end(step, {'batch': step, 'size': ANY, **train_batch_dict}))
            call_list.append(call.on_epoch_end(epoch, logs[epoch - 1]))
        call_list.append(call.on_train_end({}))

        method_calls = self.notification_callback_mock.method_calls
        self.assertIn(call.set_model(self.model), method_calls[:2])  # skip set_model and set param call
        self.assertIn(call.set_params(params), method_calls[:2])

        self.assertEqual(len(method_calls), len(call_list) + 2)  # for set_model and set param
        self.assertEqual(method_calls[2:], call_list)

    def test_givenANotificationCallback_whenTrainingLoop_thenSendNotification(self):
        notification_callback = NotificationCallback(notificator=self.notificator_mock)
        logs = self.model.fit_generator(self.train_generator,
                                        self.valid_generator,
                                        epochs=NotificationCallbackTest.epochs,
                                        steps_per_epoch=NotificationCallbackTest.steps_per_epoch,
                                        validation_steps=NotificationCallbackTest.steps_per_epoch,
                                        callbacks=[notification_callback])

        self._test_notificator_call(logs)

    def test_givenANotificationCallbackWithExperimentName_whenTrainingLoop_thenSendNotificationWithExperimentName(self):
        a_experiment_name = "A experiment name"
        notification_callback = NotificationCallback(notificator=self.notificator_mock,
                                                     experiment_name=a_experiment_name)
        logs = self.model.fit_generator(self.train_generator,
                                        self.valid_generator,
                                        epochs=NotificationCallbackTest.epochs,
                                        steps_per_epoch=NotificationCallbackTest.steps_per_epoch,
                                        validation_steps=NotificationCallbackTest.steps_per_epoch,
                                        callbacks=[notification_callback])

        self._test_notificator_call(logs, experiment_name=a_experiment_name)

    def _test_notificator_call(self, logs: Dict, experiment_name=None):
        experiment_name_text = f" for {experiment_name}" if experiment_name is not None else ""
        call_list = []
        call_list.append(call.send_notification('', subject=f'Start of the training{experiment_name_text}.'))
        for batch_log in logs:
            formatted_log_data = " ".join([f"{key}: {value}\n" for key, value in batch_log.items()])
            call_list.append(
                call.send_notification(f"Here the epoch metrics: \n{formatted_log_data}",
                                       subject=f"Epoch {batch_log['epoch']} is done{experiment_name_text}."))
        call_list.append(call.send_notification('', subject=f'End of the training{experiment_name_text}.'))

        method_calls = self.notificator_mock.method_calls
        self.assertEqual(len(method_calls), len(call_list))  # for set_model and set param
        self.assertEqual(method_calls, call_list)
