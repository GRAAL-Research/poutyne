import os

import unittest
from unittest import TestCase

from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from poutyne import torch_to_numpy
from poutyne.framework import Model
from poutyne.framework.callbacks import ModelCheckpoint


def some_data_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y

class ModelCheckpointTest(TestCase):
    batch_size = 20

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_module = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_module.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        self.checkpoint_filename = os.path.join(self.temp_dir_obj.name,
                                                'my_checkpoint_{epoch}.ckpt')

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_integration(self):
        train_gen = some_data_generator(ModelCheckpointTest.batch_size)
        valid_gen = some_data_generator(ModelCheckpointTest.batch_size)
        checkpointer = ModelCheckpoint(self.checkpoint_filename,
                                       monitor='val_loss',
                                       verbose=True,
                                       save_best_only=True)
        self.model.fit_generator(train_gen, valid_gen,
                                 epochs=10,
                                 steps_per_epoch=5,
                                 callbacks=[checkpointer])

    def test_temporary_filename_arg(self):
        tmp_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint.tmp.ckpt')
        checkpoint_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint.ckpt')
        train_gen = some_data_generator(ModelCheckpointTest.batch_size)
        valid_gen = some_data_generator(ModelCheckpointTest.batch_size)
        checkpointer = ModelCheckpoint(checkpoint_filename,
                                       monitor='val_loss',
                                       verbose=True,
                                       period=1,
                                       temporary_filename=tmp_filename)
        self.model.fit_generator(train_gen, valid_gen,
                                 epochs=10,
                                 steps_per_epoch=5,
                                 callbacks=[checkpointer])
        self.assertFalse(os.path.isfile(tmp_filename))
        self.assertTrue(os.path.isfile(checkpoint_filename))

    def test_temporary_filename_arg_with_differing_checkpoint_filename(self):
        epochs = 10
        tmp_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint.tmp.ckpt')
        checkpoint_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint_{epoch}.ckpt')
        train_gen = some_data_generator(ModelCheckpointTest.batch_size)
        valid_gen = some_data_generator(ModelCheckpointTest.batch_size)
        checkpointer = ModelCheckpoint(checkpoint_filename,
                                       monitor='val_loss',
                                       verbose=True,
                                       period=1,
                                       temporary_filename=tmp_filename)
        self.model.fit_generator(train_gen, valid_gen,
                                 epochs=epochs,
                                 steps_per_epoch=5,
                                 callbacks=[checkpointer])
        self.assertFalse(os.path.isfile(tmp_filename))
        for i in range(1, epochs+1):
            self.assertTrue(os.path.isfile(checkpoint_filename.format(epoch=i)))

    def test_non_atomic_write(self):
        checkpoint_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint.ckpt')
        train_gen = some_data_generator(ModelCheckpointTest.batch_size)
        valid_gen = some_data_generator(ModelCheckpointTest.batch_size)
        checkpointer = ModelCheckpoint(checkpoint_filename,
                                       monitor='val_loss',
                                       verbose=True,
                                       period=1,
                                       atomic_write=False)
        self.model.fit_generator(train_gen, valid_gen,
                                 epochs=10,
                                 steps_per_epoch=5,
                                 callbacks=[checkpointer])
        self.assertTrue(os.path.isfile(checkpoint_filename))

    def test_save_best_only(self):
        checkpointer = ModelCheckpoint(self.checkpoint_filename,
                                       monitor='val_loss',
                                       verbose=True,
                                       save_best_only=True)

        val_losses = [10, 3, 8, 5, 2]
        has_checkpoints = [True, True, False, False, True]
        self._test_checkpointer_with_val_losses(checkpointer, val_losses, has_checkpoints)

    def test_save_best_only_with_restore_best(self):
        checkpointer = ModelCheckpoint(self.checkpoint_filename,
                                       monitor='val_loss',
                                       verbose=True,
                                       save_best_only=True,
                                       restore_best=True)

        val_losses = [10, 3, 8, 5, 2]
        has_checkpoints = [True, True, False, False, True]
        self._test_checkpointer_with_val_losses(checkpointer, val_losses, has_checkpoints)

        self._test_restore_best(val_losses)

    def test_restore_best_without_save_best_only(self):
        with self.assertRaises(ValueError):
            ModelCheckpoint(self.checkpoint_filename,
                            monitor='val_loss',
                            verbose=True,
                            save_best_only=False,
                            restore_best=True)

        with self.assertRaises(ValueError):
            ModelCheckpoint(self.checkpoint_filename,
                            monitor='val_loss',
                            verbose=True,
                            restore_best=True)

    def test_save_best_only_with_max(self):
        checkpointer = ModelCheckpoint(self.checkpoint_filename,
                                       monitor='val_loss',
                                       mode='max',
                                       verbose=True,
                                       save_best_only=True)

        val_losses = [2, 3, 8, 5, 2]
        has_checkpoints = [True, True, True, False, False]
        self._test_checkpointer_with_val_losses(checkpointer, val_losses, has_checkpoints)

    def test_periodic_with_period_of_1(self):
        checkpointer = ModelCheckpoint(self.checkpoint_filename,
                                       monitor='val_loss',
                                       verbose=True,
                                       period=1,
                                       save_best_only=False)

        val_losses = [1] * 10
        has_checkpoints = [True] * 10
        self._test_checkpointer_with_val_losses(checkpointer, val_losses, has_checkpoints)

    def test_periodic_with_period_of_2(self):
        checkpointer = ModelCheckpoint(self.checkpoint_filename,
                                       monitor='val_loss',
                                       verbose=True,
                                       period=2,
                                       save_best_only=False)

        val_losses = [1] * 10
        has_checkpoints = [False, True] * 5
        self._test_checkpointer_with_val_losses(checkpointer, val_losses, has_checkpoints)

    def _test_checkpointer_with_val_losses(self, checkpointer, val_losses, has_checkpoints):
        generator = some_data_generator(ModelCheckpointTest.batch_size)

        checkpointer.set_params({'epochs': len(val_losses), 'steps': 1})
        checkpointer.set_model(self.model)
        checkpointer.on_train_begin({})
        for epoch, (val_loss, has_checkpoint) in enumerate(zip(val_losses, has_checkpoints), 1):
            checkpointer.on_epoch_begin(epoch, {})
            checkpointer.on_batch_begin(1, {})
            loss = self._update_model(generator)
            checkpointer.on_batch_end(
                1, {'batch': 1, 'size': ModelCheckpointTest.batch_size, 'loss': loss}
            )
            checkpointer.on_epoch_end(epoch, {'epoch': epoch, 'loss': loss, 'val_loss': val_loss})
            filename = self.checkpoint_filename.format(epoch=epoch)
            self.assertEqual(has_checkpoint, os.path.isfile(filename))
        checkpointer.on_train_end({})

    def _update_model(self, generator):
        self.pytorch_module.zero_grad()

        x, y = next(generator)
        pred_y = self.pytorch_module(x)
        loss = self.loss_function(pred_y, y)
        loss.backward()

        self.optimizer.step()

        return float(loss)

    def _test_restore_best(self, val_losses):
        final_weights = torch_to_numpy(self.model.get_weight_copies())

        epoch = val_losses.index(min(val_losses)) + 1
        best_epoch_filename = self.checkpoint_filename.format(epoch=epoch)
        self.model.load_weights(best_epoch_filename)

        best_weights = torch_to_numpy(self.model.get_weight_copies())

        self.assertEqual(best_weights, final_weights)

if __name__ == '__main__':
    unittest.main()
