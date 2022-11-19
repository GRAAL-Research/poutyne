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

# pylint: disable=too-many-locals
import os
from unittest.mock import MagicMock, call, ANY

import numpy as np
import torch
import torch.nn as nn

from poutyne import Callback
from tests.framework.base import CaptureOutputBase


class ModelFittingTestCase(CaptureOutputBase):
    epochs = 10
    steps_per_epoch = 5
    batch_size = 20

    evaluate_dataset_len = 107

    cuda_device = int(os.environ.get('CUDA_DEVICE', 0))

    def setUp(self):
        self.mock_callback = MagicMock(spec=Callback)
        self.batch_metrics = []
        self.batch_metrics_names = []
        self.batch_metrics_values = []
        self.epoch_metrics = []
        self.epoch_metrics_names = []
        self.epoch_metrics_values = []
        self.model = None
        self.pytorch_network = None
        self.optimizer = None

    def _get_callback_expected_on_calls_when_training(self, params, logs, has_valid=True, steps=None, valid_steps=None):
        # pylint: disable=too-many-arguments
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
            if has_valid:
                call_list.append(call.on_valid_begin({}))
                for step in range(1, valid_steps + 1):
                    call_list.append(call.on_valid_batch_begin(step, {}))
                    call_list.append(
                        call.on_valid_batch_end(step, {'batch': step, 'size': ANY, 'time': ANY, **val_batch_dict})
                    )
                call_list.append(call.on_valid_end({'time': ANY, **val_batch_dict, **val_epochs_dict}))
            call_list.append(call.on_epoch_end(epoch, logs[epoch - 1]))

        call_list.append(call.on_train_end({}))
        return call_list

    def _test_callbacks_train(self, params, logs, *args, **kwargs):
        self.assertEqual(len(logs), params['epochs'])

        call_list = self._get_callback_expected_on_calls_when_training(params, logs, *args, **kwargs)

        method_calls = self.mock_callback.method_calls
        self.assertIn(call.set_model(self.model), method_calls[:2])  # skip set_model and set param call
        self.assertIn(call.set_params(params), method_calls[:2])

        self.assertEqual(len(method_calls), len(call_list) + 2)  # for set_model and set param
        self.assertEqual(method_calls[2:], call_list)

    def _get_callback_expected_on_calls_when_testing(self, params):
        test_batch_dict = {"time": ANY, "test_loss": ANY}
        test_batch_dict.update(
            {
                "test_" + metric_name: metric
                for metric_name, metric in zip(self.batch_metrics_names, self.batch_metrics_values)
            }
        )

        call_list = []
        call_list.append(call.on_test_begin({}))
        for batch in range(1, params['steps'] + 1):
            call_list.append(call.on_test_batch_begin(batch, {}))
            call_list.append(call.on_test_batch_end(batch, {'batch': batch, 'size': ANY, **test_batch_dict}))

        test_batch_dict.update(
            {
                "test_" + metric_name: metric
                for metric_name, metric in zip(self.epoch_metrics_names, self.epoch_metrics_values)
            }
        )
        call_list.append(call.on_test_end({"time": ANY, "test_loss": ANY, **test_batch_dict}))
        return call_list

    def _test_callbacks_test(self, params):
        call_list = self._get_callback_expected_on_calls_when_testing(params)

        method_calls = self.mock_callback.method_calls
        self.assertEqual(call.set_model(self.model), method_calls[0])  # skip set_model and set param call
        self.assertEqual(call.set_params(params), method_calls[1])

        self.assertEqual(len(method_calls), len(call_list) + 2)  # for set_model and set param
        self.assertEqual(method_calls[2:], call_list)

    def _test_return_dict_logs(self, logs):
        test_logs = {"time": ANY, "test_loss": ANY}
        test_logs.update(
            {
                "test_" + metric_name: metric
                for metric_name, metric in zip(self.batch_metrics_names, self.batch_metrics_values)
            }
        )
        test_logs.update(
            {
                "test_" + metric_name: metric
                for metric_name, metric in zip(self.epoch_metrics_names, self.epoch_metrics_values)
            }
        )
        self.assertEqual(logs, test_logs)

    def _test_size_and_type_for_generator(self, pred_y, expected_size):
        if isinstance(pred_y, (list, tuple)):
            for o in pred_y:
                self._test_size_and_type_for_generator(o, expected_size)
        elif isinstance(pred_y, dict):
            for val in pred_y.values():
                self._test_size_and_type_for_generator(val, expected_size)
        else:
            self.assertEqual(type(pred_y), np.ndarray)
            self.assertEqual(pred_y.shape, expected_size)

    def _test_device(self, device):
        for p in self.pytorch_network.parameters():
            self.assertEqual(p.device, device)

        for p in self.optimizer.state.keys():
            if torch.is_tensor(p):
                self.assertEqual(p.device, device)

        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if torch.is_tensor(param):
                    self.assertEqual(param.device, device)

                for n, v in self.optimizer.state[param].items():
                    if 'capturable' not in param_group or param_group["capturable"] or n != 'step':
                        if torch.is_tensor(v):
                            self.assertEqual(v.device, device, n)


class MultiIOModel(nn.Module):
    """Model to test multiple inputs/outputs"""

    def __init__(self, num_input=2, num_output=2):
        super().__init__()
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
