# abstract-method because nn.Module has the abstract method _forward_unimplemented
# pylint: disable=too-many-locals,abstract-method
import io
import os
import sys
from unittest import TestCase
from unittest.mock import MagicMock, call, ANY

import numpy as np
import torch
import torch.nn as nn

from poutyne import Callback


class ModelFittingTestCase(TestCase):
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

    def _test_callbacks_train(self, params, logs, has_valid=True, steps=None):
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
                call_list.append(call.on_train_batch_begin(step, {}))
                call_list.append(call.on_backward_end(step))
                call_list.append(call.on_train_batch_end(step, {'batch': step, 'size': ANY, **train_batch_dict}))
            call_list.append(call.on_epoch_end(epoch, logs[epoch - 1]))
        call_list.append(call.on_train_end({}))

        method_calls = self.mock_callback.method_calls
        self.assertIn(call.set_model(self.model), method_calls[:2])  # skip set_model and set param call
        self.assertIn(call.set_params(params), method_calls[:2])

        self.assertEqual(len(method_calls), len(call_list) + 2)  # for set_model and set param
        self.assertEqual(method_calls[2:], call_list)

    def _test_callbacks_test(self, params):
        test_batch_dict = {"time": ANY, "test_loss": ANY}
        test_batch_dict.update({
            "test_" + metric_name: metric
            for metric_name, metric in zip(self.batch_metrics_names, self.batch_metrics_values)
        })

        call_list = []
        call_list.append(call.on_test_begin({}))
        for batch in range(1, params['steps'] + 1):
            call_list.append(call.on_test_batch_begin(batch, {}))
            call_list.append(call.on_test_batch_end(batch, {'batch': batch, 'size': ANY, **test_batch_dict}))

        test_batch_dict.update({
            "test_" + metric_name: metric
            for metric_name, metric in zip(self.epoch_metrics_names, self.epoch_metrics_values)
        })
        call_list.append(call.on_test_end({"time": ANY, "test_loss": ANY, **test_batch_dict}))

        method_calls = self.mock_callback.method_calls
        self.assertEqual(call.set_model(self.model), method_calls[0])  # skip set_model and set param call
        self.assertEqual(call.set_params(params), method_calls[1])

        self.assertEqual(len(method_calls), len(call_list) + 2)  # for set_model and set param
        self.assertEqual(method_calls[2:], call_list)

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

    def _capture_output(self):
        self.test_out = io.StringIO()
        self.original_output = sys.stdout
        sys.stdout = self.test_out


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
