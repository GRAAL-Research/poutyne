import unittest
from unittest import TestCase
from unittest.mock import MagicMock, call, ANY

import torch
import torch.nn as nn

from pytoune.framework import Model
from pytoune.framework.callbacks import DelayCallback


def some_data_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y

class DelayCallbackTest(TestCase):
    epochs = 10
    steps_per_epoch = 5
    batch_size = 20

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_module = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_module.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function)
        self.mock_callback = MagicMock()
        self.delay_callback = DelayCallback(self.mock_callback)
        self.train_dict = {'loss': ANY, 'time': ANY}
        self.log_dict = {'loss': ANY, 'val_loss': ANY, 'time': ANY}

    def test_epoch_delay(self):
        epoch_delay = 4
        delay_callback = DelayCallback(self.mock_callback, epoch_delay=epoch_delay)
        train_generator = some_data_generator(DelayCallbackTest.batch_size)
        valid_generator = some_data_generator(DelayCallbackTest.batch_size)
        self.model.fit_generator(train_generator, valid_generator,
                                 epochs=DelayCallbackTest.epochs,
                                 steps_per_epoch=DelayCallbackTest.steps_per_epoch,
                                 validation_steps=DelayCallbackTest.steps_per_epoch,
                                 callbacks=[delay_callback])
        params = {'epochs': DelayCallbackTest.epochs, 'steps': DelayCallbackTest.steps_per_epoch}

        call_list = []
        call_list.append(call.on_train_begin({}))
        for epoch in range(epoch_delay + 1, DelayCallbackTest.epochs+1):
            call_list.append(call.on_epoch_begin(epoch, {}))
            for step in range(1, params['steps']+1):
                call_list.append(call.on_batch_begin(step, {}))
                call_list.append(call.on_backward_end(step))
                call_list.append(call.on_batch_end(
                    step, {'batch': step, 'size': DelayCallbackTest.batch_size, **self.train_dict}
                ))
            call_list.append(call.on_epoch_end(epoch, {'epoch': epoch, **self.log_dict}))
        call_list.append(call.on_train_end({}))

        method_calls = self.mock_callback.method_calls
        self.assertIn(call.set_model(self.model), method_calls[:2])
        self.assertIn(call.set_params(params), method_calls[:2])

        self.assertEqual(len(method_calls), len(call_list) + 2)
        self.assertEqual(method_calls[2:], call_list)

    def test_batch_delay_in_middle_of_epoch(self):
        self._test_batch_delay(epoch_delay=5, batch_in_epoch_delay=3)

    def test_batch_delay_at_begin_of_epoch(self):
        self._test_batch_delay(epoch_delay=5, batch_in_epoch_delay=0)

    def test_batch_delay_when_no_delay(self):
        self._test_batch_delay(epoch_delay=0, batch_in_epoch_delay=0)

    def _test_batch_delay(self, epoch_delay, batch_in_epoch_delay):
        batch_delay = epoch_delay*DelayCallbackTest.steps_per_epoch + batch_in_epoch_delay
        delay_callback = DelayCallback(self.mock_callback, batch_delay=batch_delay)
        train_generator = some_data_generator(DelayCallbackTest.batch_size)
        valid_generator = some_data_generator(DelayCallbackTest.batch_size)
        self.model.fit_generator(train_generator, valid_generator,
                                 epochs=DelayCallbackTest.epochs,
                                 steps_per_epoch=DelayCallbackTest.steps_per_epoch,
                                 validation_steps=DelayCallbackTest.steps_per_epoch,
                                 callbacks=[delay_callback])
        params = {'epochs': DelayCallbackTest.epochs, 'steps': DelayCallbackTest.steps_per_epoch}

        call_list = []
        call_list.append(call.on_train_begin({}))
        for epoch in range(epoch_delay + 1, DelayCallbackTest.epochs+1):
            call_list.append(call.on_epoch_begin(epoch, {}))
            start_step = batch_in_epoch_delay + 1 if epoch == epoch_delay + 1 else 1
            for step in range(start_step, params['steps']+1):
                call_list.append(call.on_batch_begin(step, {}))
                call_list.append(call.on_backward_end(step))
                call_list.append(call.on_batch_end(
                    step, {'batch': step, 'size': DelayCallbackTest.batch_size, **self.train_dict}
                ))
            call_list.append(call.on_epoch_end(epoch, {'epoch': epoch, **self.log_dict}))
        call_list.append(call.on_train_end({}))


        method_calls = self.mock_callback.method_calls
        self.assertIn(call.set_model(self.model), method_calls[:2])
        self.assertIn(call.set_params(params), method_calls[:2])

        self.assertEqual(len(method_calls), len(call_list) + 2)
        self.assertEqual(method_calls[2:], call_list)

if __name__ == '__main__':
    unittest.main()
