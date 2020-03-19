# pylint: disable=too-many-locals
import os
from unittest import TestCase
from unittest.mock import MagicMock, call, ANY

import numpy as np
import torch
import torch.nn as nn

from poutyne.utils import _concat


class ModelFittingTestCase(TestCase):
    epochs = 10
    steps_per_epoch = 5
    batch_size = 20

    evaluate_dataset_len = 107

    cuda_device = int(os.environ.get('CUDA_DEVICE', 0))

    def setUp(self):
        self.mock_callback = MagicMock()
        self.batch_metrics = []
        self.batch_metrics_names = []
        self.batch_metrics_values = []
        self.epoch_metrics = []
        self.epoch_metrics_names = []
        self.epoch_metrics_values = []
        self.model = None

    def _test_train_fitting(self, params, logs, has_valid=True, steps=None):
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
            call_list.append(call.on_epoch_end(epoch, {'epoch': epoch, **log_dict}))
        call_list.append(call.on_train_end({}))

        method_calls = self.mock_callback.method_calls
        self.assertIn(call.set_model(self.model), method_calls[:2])  # skip set_model and set param call
        self.assertIn(call.set_params(params), method_calls[:2])

        self.assertEqual(len(method_calls), len(call_list) + 2)  # for set_model and set param
        self.assertEqual(method_calls[2:], call_list)

    def _test_test_fitting(self, params, result_log):
        test_batch_dict = dict(zip(self.batch_metrics_names, self.batch_metrics_values), loss=ANY, time=ANY)

        call_list = []
        call_list.append(call.on_test_begin({}))
        for batch in range(1, params['batch'] + 1):
            call_list.append(call.on_test_batch_begin(batch, {}))
            call_list.append(call.on_test_batch_end(batch, {'batch': batch, 'size': ANY, **test_batch_dict}))
        call_list.append(call.on_test_end(result_log))

        method_calls = self.mock_callback.method_calls
        self.assertIn(call.set_model(self.model), method_calls[:1])  # skip set_model

        self.assertEqual(len(method_calls), len(call_list) + 1)  # for set_model
        self.assertEqual(method_calls[1:], call_list)

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

    def _test_predictions_for_evaluate_and_predict_generator(self, pred_y, multi_output=False):
        self.assertEqual(type(pred_y), list)
        remaning_example = ModelFittingTestCase.evaluate_dataset_len
        cur_batch_size = ModelFittingTestCase.batch_size

        def down_the_rabbit_hole(obj, cur_batch_size):
            if isinstance(obj, (list, tuple)):
                for o in obj:
                    down_the_rabbit_hole(o, cur_batch_size)
            elif isinstance(obj, dict):
                for val in obj.values():
                    down_the_rabbit_hole(val, cur_batch_size)
            else:
                self.assertEqual(type(obj), np.ndarray)
                self.assertEqual(obj.shape, (cur_batch_size, 1))

        for pred in pred_y:
            if remaning_example < ModelFittingTestCase.batch_size:
                cur_batch_size = remaning_example
                remaning_example = 0
            else:
                remaning_example -= ModelFittingTestCase.batch_size
            self._test_size_and_type_for_generator(pred, (cur_batch_size, 1))
        if multi_output:
            for pred in _concat(pred_y):
                self.assertEqual(pred.shape, (ModelFittingTestCase.evaluate_dataset_len, 1))
        else:
            self.assertEqual(_concat(pred_y).shape, (ModelFittingTestCase.evaluate_dataset_len, 1))


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
