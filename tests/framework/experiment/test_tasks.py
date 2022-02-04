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
        self.assertEqual(expt.model.loss_function, F.cross_entropy)
        self.assertEqual(expt.model.batch_metrics, [acc])
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
        self.assertEqual(expt.model.loss_function, F.mse_loss)
        self.assertEqual(len(expt.model.batch_metrics), 0)
        self.assertEqual(len(expt.model.epoch_metrics), 0)
