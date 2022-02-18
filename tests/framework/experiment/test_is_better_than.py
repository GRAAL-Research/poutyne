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
from unittest.mock import patch, MagicMock

import pandas as pd
from torch import nn

from poutyne import Experiment, ModelBundle


class ExperimentIsBetterThanTest(TestCase):
    def setUpTwoExperiment(self, a_params=None, b_params=None):
        temp_dir_obj = TemporaryDirectory()
        test_checkpoints_path_a = os.path.join(temp_dir_obj.name, 'expt_a')
        test_checkpoints_path_b = os.path.join(temp_dir_obj.name, 'expt_b')

        if a_params is None:
            a_params = {}

        self.test_experiment_a = Experiment(test_checkpoints_path_a, nn.Linear(1, 1), **a_params)

        if b_params is None:
            b_params = {}

        self.test_experiment_b = Experiment(test_checkpoints_path_b, nn.Linear(1, 1), **b_params)

    def test_givenAIsSmallerThanBMinMonitoring_thenReturnTrue(self):
        self.setUpTwoExperiment()

        with patch.object(ModelBundle, "load_checkpoint") as load_checkpoint_mock:
            mocked_stats = MagicMock(spec=pd.DataFrame)
            series_mock = MagicMock(spec=pd.Series)
            series_mock.item.side_effect = [[1], [2]]  # The monitored metric values
            mocked_stats.__getitem__.return_value = series_mock
            load_checkpoint_mock.return_value = mocked_stats

            self.assertTrue(self.test_experiment_a.is_better_than(self.test_experiment_b))

    def test_givenAIsGreaterThanBMinMonitoring_thenReturnFalse(self):
        self.setUpTwoExperiment()

        with patch.object(ModelBundle, "load_checkpoint") as load_checkpoint_mock:
            mocked_stats = MagicMock(spec=pd.DataFrame)
            series_mock = MagicMock(spec=pd.Series)
            series_mock.item.side_effect = [[2], [1]]  # The monitored metric values
            mocked_stats.__getitem__.return_value = series_mock
            load_checkpoint_mock.return_value = mocked_stats

            self.assertFalse(self.test_experiment_a.is_better_than(self.test_experiment_b))

    def test_givenAIsSmallerThanBMaxMonitoring_thenReturnFalse(self):
        # We need to specify the metric for the mode to be properly set to "max"
        params = {"monitor_mode": "max", "monitor_metric": "loss"}
        self.setUpTwoExperiment(a_params=params, b_params=params)

        with patch.object(ModelBundle, "load_checkpoint") as load_checkpoint_mock:
            mocked_stats = MagicMock(spec=pd.DataFrame)
            series_mock = MagicMock(spec=pd.Series)
            series_mock.item.side_effect = [[1], [2]]  # The monitored metric values
            mocked_stats.__getitem__.return_value = series_mock
            load_checkpoint_mock.return_value = mocked_stats

            self.assertFalse(self.test_experiment_a.is_better_than(self.test_experiment_b))

    def test_givenAIsGreaterThanBMaxMonitoring_thenReturnTrue(self):
        # We need to specify the metric for the mode to be properly set to "max"
        params = {"monitor_mode": "max", "monitor_metric": "loss"}
        self.setUpTwoExperiment(a_params=params, b_params=params)

        with patch.object(ModelBundle, "load_checkpoint") as load_checkpoint_mock:
            mocked_stats = MagicMock(spec=pd.DataFrame)
            series_mock = MagicMock(spec=pd.Series)
            series_mock.item.side_effect = [[2], [1]]  # The monitored metric values
            mocked_stats.__getitem__.return_value = series_mock
            load_checkpoint_mock.return_value = mocked_stats

            self.assertTrue(self.test_experiment_a.is_better_than(self.test_experiment_b))

    def test_givenSomeExperimentNotLogging_thenRaiseValueError(self):
        params_a = {"logging": False}
        params_b = {"logging": True}
        self.setUpTwoExperiment(a_params=params_a, b_params=params_b)
        with self.assertRaises(ValueError):
            self.test_experiment_a.is_better_than(self.test_experiment_b)

        params_a = {"logging": True}
        params_b = {"logging": False}
        self.setUpTwoExperiment(a_params=params_a, b_params=params_b)
        with self.assertRaises(ValueError):
            self.test_experiment_a.is_better_than(self.test_experiment_b)

        params = {"logging": False}
        self.setUpTwoExperiment(a_params=params, b_params=params)
        with self.assertRaises(ValueError):
            self.test_experiment_a.is_better_than(self.test_experiment_b)

    def test_givenDifferentMonitorMetric_thenRaiseValueError(self):
        params_a = {"monitor_metric": "loss"}
        params_b = {"monitor_metric": "acc"}
        self.setUpTwoExperiment(a_params=params_a, b_params=params_b)
        with self.assertRaises(ValueError):
            self.test_experiment_a.is_better_than(self.test_experiment_b)

    def test_givenDifferentMonitorMode_thenRaiseValueError(self):
        # We need to specify the metric for the mode to be properly set to "max"
        params_a = {"monitor_mode": "max", "monitor_metric": "loss"}
        params_b = {"monitor_mode": "min"}
        self.setUpTwoExperiment(a_params=params_a, b_params=params_b)
        with self.assertRaises(ValueError):
            self.test_experiment_a.is_better_than(self.test_experiment_b)
