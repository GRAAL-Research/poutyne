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

import os
import unittest
from tempfile import TemporaryDirectory
from typing import Dict, IO
from unittest import TestCase
from unittest.mock import MagicMock, call

import torch
import torch.nn as nn

from poutyne import Model, PeriodicSaveCallback, PeriodicSaveLambda
from tests.framework.tools import some_data_generator


class PeriodicEpochSave(PeriodicSaveCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, open_mode='w', read_mode='r', **kwargs)
        self.epoch_number = None

    def save_file(self, fd: IO, epoch_number: int, logs: Dict):
        print(epoch_number, file=fd)
        self.last_saved_epoch_number = epoch_number

    def restore(self, fd: IO):
        self.restored_epoch_number = int(fd.read())


class PeriodicSaveTest(TestCase):
    batch_size = 20

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        self.save_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint_{epoch}.ckpt')

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_integration(self):
        train_gen = some_data_generator(PeriodicSaveTest.batch_size)
        valid_gen = some_data_generator(PeriodicSaveTest.batch_size)
        saver = PeriodicEpochSave(self.save_filename, monitor='val_loss', verbose=True, save_best_only=True)
        self.model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[saver])

    def test_mode_not_min_max_raise_error(self):
        with self.assertRaises(ValueError):
            invalid_mode = "a_mode"
            PeriodicEpochSave(
                self.save_filename, monitor='val_loss', verbose=True, save_best_only=True, mode=invalid_mode
            )

    def test_incorrect_monitored_metric_name_raise_key_error(self):
        invalid_monitor_metric_name = "invalid_monitor_metric_name"
        periodic_epoch_save = PeriodicEpochSave(
            self.save_filename, monitor=invalid_monitor_metric_name, verbose=True, save_best_only=True
        )

        an_epoch_number = 1
        a_log_dict = {"epoch": 1, 'acc': 1.0, 'another_metric_name': 1.0}
        with self.assertRaises(KeyError):
            periodic_epoch_save.on_epoch_end(an_epoch_number, a_log_dict)

    def test_integration_correct_monitored_metric_name(self):
        correct_monitor_metric_name = "val_loss"
        periodic_epoch_save = PeriodicEpochSave(
            self.save_filename, monitor=correct_monitor_metric_name, verbose=True, save_best_only=True
        )

        an_epoch_number = 1
        a_log_dict = {"epoch": 1, 'acc': 1.0, 'val_loss': 1.0}
        periodic_epoch_save.on_epoch_end(an_epoch_number, a_log_dict)

    def test_integration_with_keep_only_last_best(self):
        train_gen = some_data_generator(PeriodicSaveTest.batch_size)
        valid_gen = some_data_generator(PeriodicSaveTest.batch_size)
        saver = PeriodicEpochSave(
            self.save_filename, monitor='val_loss', verbose=True, save_best_only=True, keep_only_last_best=True
        )
        self.model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[saver])

    def test_temporary_filename_arg(self):
        tmp_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint.tmp.ckpt')
        save_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint.ckpt')
        train_gen = some_data_generator(PeriodicSaveTest.batch_size)
        valid_gen = some_data_generator(PeriodicSaveTest.batch_size)
        saver = PeriodicEpochSave(
            save_filename, monitor='val_loss', verbose=True, period=1, temporary_filename=tmp_filename
        )
        self.model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[saver])
        self.assertFalse(os.path.isfile(tmp_filename))
        self.assertTrue(os.path.isfile(save_filename))

    def test_temporary_filename_arg_with_differing_save_filename(self):
        epochs = 10
        tmp_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint.tmp.ckpt')
        save_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint_{epoch}.ckpt')
        train_gen = some_data_generator(PeriodicSaveTest.batch_size)
        valid_gen = some_data_generator(PeriodicSaveTest.batch_size)
        saver = PeriodicEpochSave(
            save_filename, monitor='val_loss', verbose=True, period=1, temporary_filename=tmp_filename
        )
        self.model.fit_generator(train_gen, valid_gen, epochs=epochs, steps_per_epoch=5, callbacks=[saver])
        self.assertFalse(os.path.isfile(tmp_filename))
        for i in range(1, epochs + 1):
            self.assertTrue(os.path.isfile(save_filename.format(epoch=i)))

    def test_non_atomic_write(self):
        save_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint.ckpt')
        train_gen = some_data_generator(PeriodicSaveTest.batch_size)
        valid_gen = some_data_generator(PeriodicSaveTest.batch_size)
        saver = PeriodicEpochSave(save_filename, monitor='val_loss', verbose=True, period=1, atomic_write=False)
        self.model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[saver])
        self.assertTrue(os.path.isfile(save_filename))

    def test_save_best_only(self):
        saver = PeriodicEpochSave(self.save_filename, monitor='val_loss', verbose=True, save_best_only=True)

        val_losses = [10, 3, 8, 5, 2]
        has_checkpoints = [True, True, False, False, True]
        self._test_saver_with_val_losses(saver, val_losses, has_checkpoints)

    def test_save_best_only_with_keep_only_last_best(self):
        saver = PeriodicEpochSave(
            self.save_filename, monitor='val_loss', verbose=True, save_best_only=True, keep_only_last_best=True
        )

        val_losses = [10, 3, 8, 5, 2]
        has_checkpoints = [True, True, False, False, True]
        self._test_saver_with_val_losses(saver, val_losses, has_checkpoints, keep_only_last_best=True)

    def test_save_best_only_with_max(self):
        saver = PeriodicEpochSave(self.save_filename, monitor='val_loss', mode='max', verbose=True, save_best_only=True)

        val_losses = [2, 3, 8, 5, 2]
        has_checkpoints = [True, True, True, False, False]
        self._test_saver_with_val_losses(saver, val_losses, has_checkpoints)

    def test_save_best_only_with_max_and_keep_only_last_best(self):
        saver = PeriodicEpochSave(
            self.save_filename,
            monitor='val_loss',
            mode='max',
            verbose=True,
            save_best_only=True,
            keep_only_last_best=True,
        )

        val_losses = [2, 3, 8, 5, 2]
        has_checkpoints = [True, True, True, False, False]
        self._test_saver_with_val_losses(saver, val_losses, has_checkpoints, keep_only_last_best=True)

    def test_periodic_with_period_of_1(self):
        saver = PeriodicEpochSave(self.save_filename, monitor='val_loss', verbose=True, period=1, save_best_only=False)

        val_losses = [1] * 10
        has_checkpoints = [True] * 10
        self._test_saver_with_val_losses(saver, val_losses, has_checkpoints)

    def test_periodic_with_period_of_2(self):
        saver = PeriodicEpochSave(self.save_filename, monitor='val_loss', verbose=True, period=2, save_best_only=False)

        val_losses = [1] * 10
        has_checkpoints = [False, True] * 5
        self._test_saver_with_val_losses(saver, val_losses, has_checkpoints)

    def test_keep_only_last_best_without_save_best_only(self):
        with self.assertRaises(ValueError):
            PeriodicEpochSave(
                self.save_filename, monitor='val_loss', verbose=True, save_best_only=False, keep_only_last_best=True
            )

        with self.assertRaises(ValueError):
            PeriodicEpochSave(self.save_filename, monitor='val_loss', verbose=True, keep_only_last_best=True)

    def test_save_best_only_with_restore_best(self):
        checkpointer = PeriodicEpochSave(
            self.save_filename, monitor='val_loss', verbose=True, save_best_only=True, restore_best=True
        )

        val_losses = [10, 3, 8, 7, 2, 5]
        has_checkpoints = [True, True, False, False, True, False]
        self._test_saver_with_val_losses(checkpointer, val_losses, has_checkpoints)

        self.assertEqual(5, checkpointer.restored_epoch_number)
        self.assertEqual(5, checkpointer.last_saved_epoch_number)

    def test_restore_best_without_save_best_only(self):
        with self.assertRaises(ValueError):
            PeriodicEpochSave(
                self.save_filename, monitor='val_loss', verbose=True, save_best_only=False, restore_best=True
            )

        with self.assertRaises(ValueError):
            PeriodicEpochSave(self.save_filename, monitor='val_loss', verbose=True, restore_best=True)

    def test_on_train_end_with_restore_best_without_training_raise_warning(self):
        checkpointer = PeriodicEpochSave(self.save_filename, monitor='val_loss', save_best_only=True, restore_best=True)
        a_log = {}
        with self.assertWarns(UserWarning):
            checkpointer.on_train_end(a_log)

    def _test_saver_with_val_losses(self, saver, val_losses, has_checkpoints, keep_only_last_best=False):
        generator = some_data_generator(PeriodicSaveTest.batch_size)

        best_checkpoint_filenames = []
        saver.set_params({'epochs': len(val_losses), 'steps': 1})
        saver.set_model(self.model)
        saver.on_train_begin({})
        for epoch, (val_loss, has_checkpoint) in enumerate(zip(val_losses, has_checkpoints), 1):
            saver.on_epoch_begin(epoch, {})
            saver.on_train_batch_begin(1, {})
            loss = self._update_model(generator)
            saver.on_train_batch_end(1, {'batch': 1, 'size': PeriodicSaveTest.batch_size, 'loss': loss})
            saver.on_epoch_end(epoch, {'epoch': epoch, 'loss': loss, 'val_loss': val_loss})

            filename = self.save_filename.format(epoch=epoch)
            self.assertEqual(has_checkpoint, os.path.isfile(filename))
            if has_checkpoint:
                self.assertEqual(f'{epoch}\n', open(filename, 'r', encoding='utf-8').read())
                best_checkpoint_filenames.append(os.path.realpath(filename))

        files = [os.path.realpath(os.path.join(self.temp_dir_obj.name, f)) for f in os.listdir(self.temp_dir_obj.name)]
        if keep_only_last_best:
            self.assertEqual(1, len(files))
            self.assertEqual(files[0], best_checkpoint_filenames[-1])
        else:
            best_checkpoint_filenames = set(best_checkpoint_filenames)
            self.assertEqual(len(best_checkpoint_filenames), len(files))
            self.assertEqual(best_checkpoint_filenames, set(files))

        saver.on_train_end({})

    def _update_model(self, generator):
        self.pytorch_network.zero_grad()

        x, y = next(generator)
        pred_y = self.pytorch_network(x)
        loss = self.loss_function(pred_y, y)
        loss.backward()

        self.optimizer.step()

        return float(loss)


class PeriodicSaveLambdaTest(TestCase):
    def setUp(self) -> None:
        self.a_filename = "a_file_name"

    def test_given_a_fun_save_file_use_function(self):
        a_function_mock = MagicMock()

        periodic_save_lambda = PeriodicSaveLambda(filename=self.a_filename, func=a_function_mock)
        a_file_descriptor_mock = MagicMock()
        a_epoch_number = 1
        a_log = {}

        periodic_save_lambda.save_file(a_file_descriptor_mock, a_epoch_number, a_log)
        a_function_mock.assert_has_calls([call(a_file_descriptor_mock, a_epoch_number, a_log)])

    def test_restore_use_lambda_function(self):
        a_function_mock = MagicMock()

        a_restore_mock_function = MagicMock()

        def a_restore_function(_):
            return a_restore_mock_function(2)

        periodic_save_lambda = PeriodicSaveLambda(
            filename=self.a_filename, func=a_function_mock, restore=a_restore_function
        )
        a_file_descriptor_mock = MagicMock()
        periodic_save_lambda.restore(a_file_descriptor_mock)

        a_restore_mock_function.assert_called()
        a_restore_mock_function.assert_has_calls([call(2)])


if __name__ == '__main__':
    unittest.main()
