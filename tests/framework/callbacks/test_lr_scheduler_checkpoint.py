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

# Because of the way the callbacks are generated, we have to disable linting here.
# pylint: disable=no-name-in-module
import os

import unittest
from unittest import TestCase
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from poutyne import torch_to_numpy, Model, LRSchedulerCheckpoint, ExponentialLR, ReduceLROnPlateau
from tests.framework.tools import some_data_generator


class OptimizerCheckpointTest(TestCase):
    batch_size = 20
    epochs = 10

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.pytorch_network.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        self.checkpoint_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint_{epoch}.optim')

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_any_scheduler_integration(self):
        train_gen = some_data_generator(OptimizerCheckpointTest.batch_size)
        valid_gen = some_data_generator(OptimizerCheckpointTest.batch_size)
        lr_scheduler = ExponentialLR(gamma=0.01)
        checkpointer = LRSchedulerCheckpoint(lr_scheduler, self.checkpoint_filename, period=1)
        self.model.fit_generator(
            train_gen, valid_gen, epochs=OptimizerCheckpointTest.epochs, steps_per_epoch=5, callbacks=[checkpointer]
        )

    def test_reduce_lr_on_plateau_integration(self):
        train_gen = some_data_generator(OptimizerCheckpointTest.batch_size)
        valid_gen = some_data_generator(OptimizerCheckpointTest.batch_size)
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=3)
        checkpointer = LRSchedulerCheckpoint(reduce_lr, self.checkpoint_filename, period=1)
        self.model.fit_generator(
            train_gen, valid_gen, epochs=OptimizerCheckpointTest.epochs, steps_per_epoch=5, callbacks=[checkpointer]
        )

    def test_any_scheduler_checkpoints(self):
        lr_scheduler = ExponentialLR(gamma=0.01)
        checkpointer = LRSchedulerCheckpoint(lr_scheduler, self.checkpoint_filename, period=1)
        self._test_checkpointer(checkpointer, lr_scheduler)

    def test_reduce_lr_checkpoints(self):
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=3)
        checkpointer = LRSchedulerCheckpoint(reduce_lr, self.checkpoint_filename, period=1)
        self._test_checkpointer(checkpointer, reduce_lr)

    def _test_checkpointer(self, checkpointer, lr_scheduler):
        scheduler_states = {}
        generator = some_data_generator(OptimizerCheckpointTest.batch_size)

        checkpointer.set_params({'epochs': OptimizerCheckpointTest.epochs, 'steps': 1})
        checkpointer.set_model(self.model)
        checkpointer.on_train_begin({})
        for epoch in range(1, OptimizerCheckpointTest.epochs + 1):
            checkpointer.on_epoch_begin(epoch, {})
            checkpointer.on_train_batch_begin(1, {})
            loss = self._update_model(generator)
            checkpointer.on_train_batch_end(1, {'batch': 1, 'size': OptimizerCheckpointTest.batch_size, 'loss': loss})
            checkpointer.on_epoch_end(epoch, {'epoch': epoch, 'loss': loss, 'val_loss': 1})
            filename = self.checkpoint_filename.format(epoch=epoch)
            self.assertTrue(os.path.isfile(filename))
            scheduler_states[epoch] = torch_to_numpy(lr_scheduler.scheduler.state_dict(), copy=True)
        checkpointer.on_train_end({})

        self._test_checkpoint(scheduler_states, lr_scheduler)

    def _update_model(self, generator):
        self.pytorch_network.zero_grad()

        x, y = next(generator)
        pred_y = self.pytorch_network(x)
        loss = self.loss_function(pred_y, y)
        loss.backward()

        self.optimizer.step()

        return float(loss)

    def _test_checkpoint(self, scheduler_states, lr_scheduler):
        for epoch, epoch_scheduler_state in scheduler_states.items():
            filename = self.checkpoint_filename.format(epoch=epoch)
            lr_scheduler.load_state(filename)
            saved_scheduler_state = torch_to_numpy(lr_scheduler.scheduler.state_dict())

            self.assertEqual(epoch_scheduler_state, saved_scheduler_state)


if __name__ == '__main__':
    unittest.main()
