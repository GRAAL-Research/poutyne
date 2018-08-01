import os

import unittest
from unittest import TestCase
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from pytoune import torch_to_numpy
from pytoune.framework import Model
from pytoune.framework.callbacks import OptimizerCheckpoint


def some_data_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y

class OptimizerCheckpointTest(TestCase):
    batch_size = 20
    epochs = 10

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_module = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.pytorch_module.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        self.checkpoint_filename = os.path.join(self.temp_dir_obj.name,
                                                'my_checkpoint_{epoch}.optim')

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_integration(self):
        train_gen = some_data_generator(OptimizerCheckpointTest.batch_size)
        valid_gen = some_data_generator(OptimizerCheckpointTest.batch_size)
        checkpointer = OptimizerCheckpoint(self.checkpoint_filename, period=1)
        self.model.fit_generator(train_gen, valid_gen,
                                 epochs=OptimizerCheckpointTest.epochs,
                                 steps_per_epoch=5,
                                 callbacks=[checkpointer])

    def test_checkpoints(self):
        checkpointer = OptimizerCheckpoint(self.checkpoint_filename, period=1)
        self._test_checkpointer(checkpointer)

    def _test_checkpointer(self, checkpointer):
        optimizer_states = {}
        generator = some_data_generator(OptimizerCheckpointTest.batch_size)

        checkpointer.set_params({'epochs': OptimizerCheckpointTest.epochs, 'steps': 1})
        checkpointer.set_model(self.model)
        checkpointer.on_train_begin({})
        for epoch in range(1, OptimizerCheckpointTest.epochs + 1):
            checkpointer.on_epoch_begin(epoch, {})
            checkpointer.on_batch_begin(1, {})
            loss = self._update_model(generator)
            checkpointer.on_batch_end(
                1, {'batch': 1, 'size': OptimizerCheckpointTest.batch_size, 'loss': loss}
            )
            checkpointer.on_epoch_end(epoch, {'epoch': epoch, 'loss': loss, 'val_loss': 1})
            filename = self.checkpoint_filename.format(epoch=epoch)
            self.assertTrue(os.path.isfile(filename))
            optimizer_states[epoch] = torch_to_numpy(self.optimizer.state_dict(), copy=True)
        checkpointer.on_train_end({})

        self._test_checkpoint(optimizer_states)

    def _update_model(self, generator):
        self.pytorch_module.zero_grad()

        x, y = next(generator)
        pred_y = self.pytorch_module(x)
        loss = self.loss_function(pred_y, y)
        loss.backward()

        self.optimizer.step()

        return float(loss)

    def _test_checkpoint(self, optimizer_states):
        for epoch, epoch_optimizer_state in optimizer_states.items():
            filename = self.checkpoint_filename.format(epoch=epoch)
            self.model.load_optimizer_state(filename)
            saved_optimizer_state = torch_to_numpy(self.optimizer.state_dict())

            self.assertEqual(epoch_optimizer_state, saved_optimizer_state)

if __name__ == '__main__':
    unittest.main()
