from unittest import TestCase
from unittest.mock import MagicMock, call, ANY, DEFAULT
from utils import CopyingMock

from pitoune.framework import Model
from pitoune.framework.callbacks import Callback, DelayCallback

import torch
import torch.nn as nn

def some_data_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y

class DelayCallbackTest(TestCase):
    n_epochs = 10
    steps_per_epoch = 5

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_module = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_module.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function)
        self.mock_callback = MagicMock()
        self.mock_callback.on_train_begin = CopyingMock()
        self.mock_callback.on_train_end = CopyingMock()
        self.mock_callback.on_epoch_begin = CopyingMock()
        self.mock_callback.on_epoch_end = CopyingMock()
        self.mock_callback.on_batch_begin = CopyingMock()
        self.mock_callback.on_batch_end = CopyingMock()
        self.delay_callback = DelayCallback(self.mock_callback)

    def test_epoch_delay(self):
        epoch_delay = 4
        delay_callback = DelayCallback(self.mock_callback, epoch_delay=epoch_delay)
        train_generator = some_data_generator(20)
        valid_generator = some_data_generator(20)
        logs = self.model.fit_generator(train_generator, valid_generator, n_epochs=DelayCallbackTest.n_epochs, steps_per_epoch=DelayCallbackTest.steps_per_epoch, callbacks=[delay_callback])
        params = {'n_epochs': DelayCallbackTest.n_epochs, 'steps_per_epoch': DelayCallbackTest.steps_per_epoch, 'metrics': []}

        method_calls = self.mock_callback.method_calls
        self.assertEqual(method_calls.index(call.on_train_begin([])), 2)
        self.assertIn(call.set_model(self.model), method_calls[:2])
        self.assertIn(call.set_params(params), method_calls[:2])

        i = 3
        for epoch in range(epoch_delay + 1, DelayCallbackTest.n_epochs+1):
            self.assertEqual(method_calls[i], call.on_epoch_begin(epoch, ANY))
            self.assertEqual(len(method_calls[i][1][1]), epoch - 1)
            i += 1
            for step in range(1, params['steps_per_epoch']+1):
                self.assertEqual(method_calls[i], call.on_batch_begin(step, ANY))
                self.assertEqual(len(method_calls[i][1][1]), epoch)
                i += 1
                self.assertEqual(method_calls[i], call.on_batch_end(step, ANY))
                self.assertEqual(len(method_calls[i][1][1]), epoch)
                i += 1

            self.assertEqual(method_calls[i], call.on_epoch_end(epoch, ANY))
            self.assertEqual(len(method_calls[i][1][1]), epoch)
            i += 1

        self.assertEqual(method_calls[i], call.on_train_end(ANY))
        self.assertEqual(len(method_calls[i][1][0]), epoch)
        i += 1

        self.assertEqual(len(method_calls), i)

    def test_batch_delay_in_middle_of_epoch(self):
        self._test_batch_delay(epoch_delay=5, batch_in_epoch_delay=3)

    def test_batch_delay_at_begin_of_epoch(self):
        self._test_batch_delay(epoch_delay=5, batch_in_epoch_delay=0)

    def _test_batch_delay(self, epoch_delay, batch_in_epoch_delay):
        batch_delay = epoch_delay*DelayCallbackTest.steps_per_epoch + batch_in_epoch_delay
        delay_callback = DelayCallback(self.mock_callback, batch_delay=batch_delay)
        train_generator = some_data_generator(20)
        valid_generator = some_data_generator(20)
        logs = self.model.fit_generator(train_generator, valid_generator, n_epochs=DelayCallbackTest.n_epochs, steps_per_epoch=DelayCallbackTest.steps_per_epoch, callbacks=[delay_callback])
        params = {'n_epochs': DelayCallbackTest.n_epochs, 'steps_per_epoch': DelayCallbackTest.steps_per_epoch, 'metrics': []}

        method_calls = self.mock_callback.method_calls
        self.assertEqual(method_calls.index(call.on_train_begin([])), 2)
        self.assertIn(call.set_model(self.model), method_calls[:2])
        self.assertIn(call.set_params(params), method_calls[:2])

        i = 3
        for epoch in range(epoch_delay + 1, DelayCallbackTest.n_epochs+1):
            self.assertEqual(method_calls[i], call.on_epoch_begin(epoch, ANY))
            len_logs = epoch if epoch == epoch_delay + 1 else epoch - 1
            self.assertEqual(len(method_calls[i][1][1]), len_logs)
            i += 1

            start_step = batch_in_epoch_delay + 1 if epoch == epoch_delay + 1 else 1
            for step in range(start_step, params['steps_per_epoch']+1):
                self.assertEqual(method_calls[i], call.on_batch_begin(step, ANY))
                self.assertEqual(len(method_calls[i][1][1]), epoch)
                i += 1
                self.assertEqual(method_calls[i], call.on_batch_end(step, ANY))
                self.assertEqual(len(method_calls[i][1][1]), epoch)
                i += 1

            self.assertEqual(method_calls[i], call.on_epoch_end(epoch, ANY))
            self.assertEqual(len(method_calls[i][1][1]), epoch)
            i += 1

        self.assertEqual(method_calls[i], call.on_train_end(ANY))
        self.assertEqual(len(method_calls[i][1][0]), epoch)
        i += 1

        self.assertEqual(len(method_calls), i)

if __name__ == '__main__':
    unittest.main()
