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
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch
import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset

# pylint: disable=no-name-in-module
from poutyne import Experiment, ExponentialLR
from tests.framework.tools import SomeDataGeneratorWithLen
from tests.framework.experiment.utils import ConstantMetric, ConstantMetricCallback


class BaseExperimentTest:
    # pylint: disable=no-member
    NUM_EPOCHS = None
    METRIC_VALUES = None
    CHECKPOINT_EPOCHS = None
    NO_CHECKPOINT_EPOCHS = None
    MONITOR_MODE = None

    def setUp(self):
        self.temp_dir_obj = TemporaryDirectory()
        self.test_checkpoints_path = os.path.join(self.temp_dir_obj.name, 'expt')

        self.metric = ConstantMetric()
        self.metric_callback = ConstantMetricCallback(self.METRIC_VALUES, self.metric)

        self.test_experiment = Experiment(
            self.test_checkpoints_path,
            nn.Linear(1, 1),
            optimizer='sgd',
            loss_function='mse',
            batch_metrics=[self.metric],
            monitor_metric="const",
            monitor_mode=self.MONITOR_MODE,
        )
        self.checkpoint_paths = [
            os.path.join(self.test_checkpoints_path, f"checkpoint_epoch_{epoch}.ckpt")
            for epoch in self.CHECKPOINT_EPOCHS
        ]
        self.no_checkpoint_paths = [
            os.path.join(self.test_checkpoints_path, f"checkpoint_epoch_{epoch}.ckpt")
            for epoch in self.NO_CHECKPOINT_EPOCHS
        ]
        self.ckpt_last_path = os.path.join(self.test_checkpoints_path, "checkpoint.ckpt")
        self.optim_ckpt_path = os.path.join(self.test_checkpoints_path, "checkpoint.optim")
        self.rng_ckpt_path = os.path.join(self.test_checkpoints_path, "checkpoint.randomstate")
        self.first_lr_scheduler_ckpt_path = os.path.join(self.test_checkpoints_path, "lr_sched_0.lrsched")
        self.second_lr_scheduler_ckpt_path = os.path.join(self.test_checkpoints_path, "lr_sched_1.lrsched")
        self.tsv_log_path = os.path.join(self.test_checkpoints_path, "log.tsv")
        self.tsv_test_log_path = os.path.join(self.test_checkpoints_path, "test_log.tsv")
        self.epoch_file_path = os.path.join(self.test_checkpoints_path, "last.epoch")
        self.time_metric_plot_png_file_path = os.path.join(self.test_checkpoints_path, "plots", "time.png")
        self.time_metric_plot_pdf_file_path = os.path.join(self.test_checkpoints_path, "plots", "time.pdf")
        self.loss_metric_plot_png_file_path = os.path.join(self.test_checkpoints_path, "plots", "loss.png")
        self.loss_metric_plot_pdf_file_path = os.path.join(self.test_checkpoints_path, "plots", "loss.pdf")

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_integration_train(self):
        train_generator = SomeDataGeneratorWithLen(32, 10, 0)
        valid_generator = SomeDataGeneratorWithLen(32, 10, 0)
        logs = self.test_experiment.train(
            train_generator, valid_generator, epochs=self.NUM_EPOCHS, callbacks=[self.metric_callback]
        )
        self._test_train_integration(logs)

    def test_integration_train_with_lr_scheduler(self):
        lr_scheduler = ExponentialLR(0.9)

        train_generator = SomeDataGeneratorWithLen(32, 10, 0)
        valid_generator = SomeDataGeneratorWithLen(32, 10, 0)
        logs = self.test_experiment.train(
            train_generator,
            valid_generator,
            epochs=self.NUM_EPOCHS,
            callbacks=[self.metric_callback],
            lr_schedulers=[lr_scheduler],
        )
        self._test_train_integration(logs, num_lr_schedulers=1)

    def test_integration_train_with_two_lr_schedulers(self):
        lr_schedulers = [ExponentialLR(0.9), ExponentialLR(0.8)]

        train_generator = SomeDataGeneratorWithLen(32, 10, 0)
        valid_generator = SomeDataGeneratorWithLen(32, 10, 0)
        logs = self.test_experiment.train(
            train_generator,
            valid_generator,
            epochs=self.NUM_EPOCHS,
            callbacks=[self.metric_callback],
            lr_schedulers=lr_schedulers,
        )
        self._test_train_integration(logs, num_lr_schedulers=2)

    def test_integration_train_dataset(self):
        train_dataset = TensorDataset(torch.rand(32 * 10, 1), torch.rand(32 * 10, 1))
        valid_dataset = TensorDataset(torch.rand(32 * 10, 1), torch.rand(32 * 10, 1))
        logs = self.test_experiment.train_dataset(
            train_dataset, valid_dataset, epochs=self.NUM_EPOCHS, callbacks=[self.metric_callback]
        )
        self._test_train_integration(logs)

    def test_integration_train_data(self):
        train_x, train_y = torch.rand(32 * 10, 1), torch.rand(32 * 10, 1)
        validation_data = torch.rand(32 * 10, 1), torch.rand(32 * 10, 1)
        logs = self.test_experiment.train_data(
            train_x, train_y, validation_data, epochs=self.NUM_EPOCHS, callbacks=[self.metric_callback]
        )
        self._test_train_integration(logs)

    def test_train_resume(self):
        train_generator = SomeDataGeneratorWithLen(32, 10, 0)
        valid_generator = SomeDataGeneratorWithLen(32, 10, 0)
        lr_schedulers = [ExponentialLR(0.9), ExponentialLR(0.8)]
        logs = self.test_experiment.train(
            train_generator,
            valid_generator,
            epochs=self.NUM_EPOCHS,
            callbacks=[self.metric_callback],
            lr_schedulers=lr_schedulers,
        )
        self._test_train_integration(logs, num_lr_schedulers=2)

        epochs = self.NUM_EPOCHS + 10
        logs = self.test_experiment.train(train_generator, valid_generator, epochs=epochs, lr_schedulers=lr_schedulers)
        self._test_train_integration(logs, epochs=epochs, initial_epoch=self.NUM_EPOCHS + 1, num_lr_schedulers=2)

    def test_train_no_log(self):
        self._test_train_no_log(0)

    def test_train_no_log_with_lr_scheduler(self):
        self._test_train_no_log(1)

    def test_train_no_log_with_two_lr_schedulers(self):
        self._test_train_no_log(2)

    def _test_train_no_log(self, num_lr_schedulers):
        lr_schedulers = None
        if num_lr_schedulers >= 1:
            lr_schedulers = [ExponentialLR(0.9 - 0.1 * i) for i in range(num_lr_schedulers)]
        test_experiment = Experiment(
            self.test_checkpoints_path,
            nn.Linear(1, 1),
            optimizer='sgd',
            loss_function='mse',
            batch_metrics=[self.metric],
            monitor_metric="const",
            monitor_mode=self.MONITOR_MODE,
            logging=False,
        )
        train_generator = SomeDataGeneratorWithLen(32, 10, 0)
        valid_generator = SomeDataGeneratorWithLen(32, 10, 0)
        logs = test_experiment.train(
            train_generator,
            valid_generator,
            epochs=self.NUM_EPOCHS,
            callbacks=[self.metric_callback],
            lr_schedulers=lr_schedulers,
        )

        self.assertFalse(os.path.isdir(self.test_checkpoints_path))
        for path in self.checkpoint_paths:
            self.assertFalse(os.path.isfile(path))
        for path in self.no_checkpoint_paths:
            self.assertFalse(os.path.isfile(path))
        self.assertFalse(os.path.isfile(self.ckpt_last_path))
        self.assertFalse(os.path.isfile(self.optim_ckpt_path))
        self.assertFalse(os.path.isfile(self.rng_ckpt_path))
        self.assertFalse(os.path.isfile(self.tsv_log_path))
        self.assertFalse(os.path.isfile(self.epoch_file_path))
        self.assertFalse(os.path.isfile(self.tsv_test_log_path))
        self.assertFalse(os.path.isfile(self.time_metric_plot_png_file_path))
        self.assertFalse(os.path.isfile(self.time_metric_plot_pdf_file_path))
        self.assertFalse(os.path.isfile(self.loss_metric_plot_png_file_path))
        self.assertFalse(os.path.isfile(self.loss_metric_plot_pdf_file_path))
        self.assertFalse(os.path.isfile(self.first_lr_scheduler_ckpt_path))
        self.assertFalse(os.path.isfile(self.second_lr_scheduler_ckpt_path))

        self.assertEqual(len(logs), self.NUM_EPOCHS)
        for i, log in enumerate(logs, 1):
            self.assertIn('epoch', log)
            self.assertEqual(log['epoch'], i)
            self.assertIn('loss', log)
            self.assertIn('val_loss', log)
            self.assertIn('time', log)

    def _test_train_integration(self, logs, epochs=None, initial_epoch=1, num_lr_schedulers=0):
        if epochs is None:
            epochs = self.NUM_EPOCHS

        self.assertTrue(os.path.isdir(self.test_checkpoints_path))
        for path in self.checkpoint_paths:
            self.assertTrue(os.path.isfile(path))
        for path in self.no_checkpoint_paths:
            self.assertFalse(os.path.isfile(path))
        self.assertTrue(os.path.isfile(self.ckpt_last_path))
        self.assertTrue(os.path.isfile(self.optim_ckpt_path))
        self.assertTrue(os.path.isfile(self.rng_ckpt_path))
        self.assertTrue(os.path.isfile(self.tsv_log_path))
        self.assertTrue(os.path.isfile(self.epoch_file_path))
        self.assertTrue(os.path.isfile(self.time_metric_plot_png_file_path))
        self.assertTrue(os.path.isfile(self.time_metric_plot_pdf_file_path))
        self.assertTrue(os.path.isfile(self.loss_metric_plot_png_file_path))
        self.assertTrue(os.path.isfile(self.loss_metric_plot_pdf_file_path))
        self.assertFalse(os.path.isfile(self.tsv_test_log_path))
        if num_lr_schedulers >= 1:
            self.assertTrue(os.path.isfile(self.first_lr_scheduler_ckpt_path))
        else:
            self.assertFalse(os.path.isfile(self.first_lr_scheduler_ckpt_path))
        if num_lr_schedulers >= 2:
            self.assertTrue(os.path.isfile(self.second_lr_scheduler_ckpt_path))
        else:
            self.assertFalse(os.path.isfile(self.second_lr_scheduler_ckpt_path))

        self.assertEqual(len(logs), epochs - initial_epoch + 1)
        for i, log in enumerate(logs, initial_epoch):
            self.assertIn('epoch', log)
            self.assertEqual(log['epoch'], i)
            self.assertIn('loss', log)
            self.assertIn('val_loss', log)
            self.assertIn('time', log)

        with open(self.epoch_file_path, 'r', encoding='utf-8') as fd:
            epoch = int(fd.read())
        self.assertEqual(epoch, epochs)

        actual_stats = pd.read_csv(self.tsv_log_path, sep='\t')
        pd.testing.assert_frame_equal(self.test_experiment.get_stats(), actual_stats)

        actual_logs = actual_stats.drop(columns=['lr'])
        actual_logs = actual_logs[actual_logs['epoch'] >= initial_epoch].reset_index(drop=True)
        expected_logs = pd.DataFrame(logs)
        pd.testing.assert_frame_equal(expected_logs, actual_logs)

    def test_get_saved_epochs(self):
        train_generator = SomeDataGeneratorWithLen(32, 10, 0)
        valid_generator = SomeDataGeneratorWithLen(32, 10, 0)
        self.test_experiment.train(
            train_generator, valid_generator, epochs=self.NUM_EPOCHS, callbacks=[self.metric_callback]
        )
        actual_saved_epochs = self.test_experiment.get_saved_epochs()

        self.assertEqual(self.CHECKPOINT_EPOCHS, actual_saved_epochs['epoch'].tolist())

        stats = self.test_experiment.get_stats()
        expected_saved_epochs = stats[stats['epoch'].isin(self.CHECKPOINT_EPOCHS)]
        pd.testing.assert_frame_equal(expected_saved_epochs, actual_saved_epochs)

    def test_integration_test(self):
        self._train_expt()
        generator = SomeDataGeneratorWithLen(32, 10, 0)
        logs = self.test_experiment.test(generator)
        self._test_test_integration(logs)

    def test_integration_test_dataset(self):
        self._train_expt()
        dataset = TensorDataset(torch.rand(32 * 10, 1), torch.rand(32 * 10, 1))
        logs = self.test_experiment.test_dataset(dataset)
        self._test_test_integration(logs)

    def test_integration_test_data(self):
        self._train_expt()
        x, y = torch.rand(32 * 10, 1), torch.rand(32 * 10, 1)
        log = self.test_experiment.test_data(x, y)
        self._test_test_integration(log)

    def test_test_with_return_dict_format_to_False_raises_exception(self):
        self._train_expt()
        generator = SomeDataGeneratorWithLen(32, 10, 0)
        with self.assertRaises(ValueError):
            self.test_experiment.test(generator, return_dict_format=False)

    def _test_test_integration(self, log):
        self.assertTrue(os.path.isfile(self.tsv_test_log_path))

        self.assertIn('test_loss', log)
        self.assertIn('time', log)

    def test_infer(self):
        self._train_expt()
        num_steps = 10
        generator = SomeDataGeneratorWithLen(32, 10, 0)
        generator = (x for x, _ in generator)
        pred_y = self.test_experiment.infer(generator, steps=num_steps)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (320, 1))

    def test_infer_dataset(self):
        self._train_expt()
        dataset = TensorDataset(torch.rand(32 * 10, 1))
        pred_y = self.test_experiment.infer_dataset(dataset)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (320, 1))

    def test_infer_data(self):
        self._train_expt()
        x = torch.rand(32 * 10, 1)
        pred_y = self.test_experiment.infer_data(x)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (320, 1))

    def test_infer_data_without_training(self):
        x = torch.rand(32 * 10, 1)
        with self.assertRaises(ValueError):
            self.test_experiment.infer_data(x)

    def _train_expt(self):
        train_generator = SomeDataGeneratorWithLen(32, 10, 0)
        valid_generator = SomeDataGeneratorWithLen(32, 10, 0)
        self.test_experiment.train(
            train_generator, valid_generator, epochs=self.NUM_EPOCHS, callbacks=[self.metric_callback]
        )


class MonitorMinExperimentTest(BaseExperimentTest, TestCase):
    NUM_EPOCHS = 5
    METRIC_VALUES = [9, 3, 6, 2, 3]
    CHECKPOINT_EPOCHS = [1, 2, 4]
    NO_CHECKPOINT_EPOCHS = [3, 5]
    MONITOR_MODE = "min"


class MonitorMaxExperimentTest(BaseExperimentTest, TestCase):
    NUM_EPOCHS = 5
    METRIC_VALUES = [4, 3, 6, 2, 7]
    CHECKPOINT_EPOCHS = [1, 3, 5]
    NO_CHECKPOINT_EPOCHS = [2, 4]
    MONITOR_MODE = "max"
