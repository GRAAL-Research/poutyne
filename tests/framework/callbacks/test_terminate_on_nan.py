import numpy as np
import torch
import torch.nn as nn

from poutyne import Model
from poutyne import TerminateOnNaN
from tests.framework.base import CaptureOutputBase
from tests.framework.tools import some_data_generator


class TerminateOnNaNTest(CaptureOutputBase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.pytorch_network = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 1))
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

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

        self.assertStdoutContains([f"Batch {a_batch_number:d}: Invalid loss, terminating training"])

    def test_without_nan_during_train_does_not_stop_training(self):
        terminate_on_nan = TerminateOnNaN()
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)

        self.model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[terminate_on_nan])
        self.assertFalse(self.model.stop_training)
