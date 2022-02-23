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

from torch import nn
import torch.nn.functional as F

from poutyne import Experiment, acc, FBeta


class ExperimentTasksTest(TestCase):
    def setUp(self):
        self.temp_dir_obj = TemporaryDirectory()
        self.test_checkpoints_path = os.path.join(self.temp_dir_obj.name, 'expt')

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_task_classif(self):
        expt = Experiment(
            self.test_checkpoints_path,
            nn.Linear(1, 10),
            optimizer='sgd',
            task='classif',
        )
        self._test_task_classif(expt)

        expt = Experiment(
            self.test_checkpoints_path,
            nn.Linear(1, 10),
            optimizer='sgd',
            task='classification',
        )
        self._test_task_classif(expt)

    def _test_task_classif(self, expt):
        self.assertEqual(expt.monitor_metric, 'val_acc')
        self.assertEqual(expt.monitor_mode, 'max')
        self.assertEqual(expt.model.loss_function.func, F.cross_entropy)
        self.assertEqual(len(expt.model.batch_metrics), 1)
        self.assertEqual(expt.model.batch_metrics[0].func, acc)
        self.assertEqual(len(expt.model.epoch_metrics), 1)
        self.assertIsInstance(expt.model.epoch_metrics[0], FBeta)

    def test_task_regr(self):
        expt = Experiment(
            self.test_checkpoints_path,
            nn.Linear(1, 10),
            optimizer='sgd',
            task='regr',
        )
        self._test_task_regr(expt)

        expt = Experiment(
            self.test_checkpoints_path,
            nn.Linear(1, 10),
            optimizer='sgd',
            task='regression',
        )
        self._test_task_regr(expt)

    def _test_task_regr(self, expt):
        self.assertEqual(expt.monitor_metric, 'val_loss')
        self.assertEqual(expt.monitor_mode, 'min')
        self.assertEqual(expt.model.loss_function.func, F.mse_loss)
        self.assertEqual(len(expt.model.batch_metrics), 0)
        self.assertEqual(len(expt.model.epoch_metrics), 0)
