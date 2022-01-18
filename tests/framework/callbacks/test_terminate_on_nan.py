import io
import sys
from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn

from poutyne import Model

from poutyne import TerminateOnNaN


# todo change to CaptureOutputBase

# class TerminateOnNaNTest(CaptureOutputBase):
class TerminateOnNaNTest(TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.pytorch_network = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 1))
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

    def _capture_output(self):
        self.test_out = io.StringIO()
        self.original_output = sys.stdout
        sys.stdout = self.test_out

    def assertStdoutContains(self, values):
        for value in values:
            self.assertIn(value, self.test_out.getvalue().strip())

    def test_on_nan_during_train_stop_training(self):
        terminate_on_nan = TerminateOnNaN()
        terminate_on_nan.set_model(self.model)

        a_batch_number = 1
        a_loss_logs_with_nan = {"loss": np.array([np.nan])}
        terminate_on_nan.on_train_batch_end(batch_number=a_batch_number, logs=a_loss_logs_with_nan)
        self.assertTrue(self.model.stop_training)

    def test_on_nan_during_train_print_error(self):
        terminate_on_nan = TerminateOnNaN()
        terminate_on_nan.set_model(self.model)

        a_batch_number = 1
        a_loss_logs_with_nan = {"loss": np.array([np.nan])}
        self._capture_output()
        terminate_on_nan.on_train_batch_end(batch_number=a_batch_number, logs=a_loss_logs_with_nan)

        self.assertStdoutContains(f"Batch {a_batch_number:d}: Invalid loss, terminating training")

    def test_without_nan_during_train_does_not_stop_training(self):
        terminate_on_nan = TerminateOnNaN()
        terminate_on_nan.set_model(self.model)

        a_batch_number = 1
        a_loss_logs_without_nan = {"loss": np.array([2.0])}
        terminate_on_nan.on_train_batch_end(batch_number=a_batch_number, logs=a_loss_logs_without_nan)
        with self.assertRaises(AttributeError):
            getattr(self.model, "stop_training")
