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
from typing import List, Dict
from typing import Mapping, Sequence
from unittest import TestCase
from unittest.mock import patch, MagicMock, call

import git
import torch
import torch.nn as nn
from mlflow.exceptions import MlflowException
from omegaconf import DictConfig

from poutyne import Model
from poutyne.framework.callbacks.mlflow_logger import _get_git_commit, MLFlowLogger
from tests.framework.tools import SomeDataGeneratorWithLen

a_git_commit = "9bff900c30e80c3a35388d3e617db5b7a64c9afd"
mlflow_default_git_commit_tag = "mlflow.source.git.commit"


class MLFlowLoggerTest(TestCase):
    def setUp(self) -> None:
        self.a_experiment_name = "a_name"
        self.a_run_id = "101010"
        self.a_experiment_id = "112321"
        self.none_tracking_uri = None

        self.a_exception_message = "a error message"

        self.the_working_directory = os.getcwd()

        self.experiment_mock = MagicMock()
        self.experiment_mock.return_value = self.a_experiment_id

        self.run_mock = MagicMock()
        self.run_mock.return_value.info.run_id = self.a_run_id

        self.num_epochs = 2
        self.steps_per_epoch = 5
        self.batch_size = 32
        self.a_log = {"metric_1": 1, "metric_2": 2}

        self.settings_in_dict = {"param_1": 1, "param_2": 2, "param_3": "value"}
        self.settings_in_dict_config_no_sequence = DictConfig(
            {'param_dict': {'param_1': 1}, 'param_dict_2': {'param_2"': 2, 'param_3"': 3}, 'param': 'value'}
        )
        self.settings_in_dict_config_with_sequence = DictConfig(
            {
                'param_dict': {'param_1': 1},
                'param_dict_2': {'param_2"': 2, 'param_3"': 3},
                'param': 'value',
                'a_list_param': [0, 1],
            }
        )

    @patch("poutyne.framework.mlflow_logger.mlflow", None)
    def test_whenMLFowNotInstalled_thenRaiseImportError(self):
        with self.assertRaises(ImportError):
            MLFlowLogger(self.a_experiment_name)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenNewExperiment_givenAMLFlowInstantiation_thenCreateNewExperiment(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock

            MLFlowLogger(self.a_experiment_name)

            create_experiment_call = [call().create_experiment(self.a_experiment_name, self.none_tracking_uri)]

            ml_flow_client_patch.assert_has_calls(create_experiment_call)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenExperimentAlreadyCreated_givenAMLFlowInstantiation_thenGetExperiment(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = MagicMock(
                side_effect=MlflowException(self.a_exception_message)
            )
            ml_flow_client_patch.return_value.get_experiment_by_name = MagicMock(return_value=self.experiment_mock)

            MLFlowLogger(self.a_experiment_name)

            create_experiment_calls = [
                call().create_experiment(self.a_experiment_name, self.none_tracking_uri),
                call().get_experiment_by_name(self.a_experiment_name),
            ]

            ml_flow_client_patch.assert_has_calls(create_experiment_calls)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenCorrectSettings_givenAMLFlowInstantiation_thenMLFlowClientIsProperlySet(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name)

            settings_calls = [
                call().create_experiment(self.a_experiment_name, self.none_tracking_uri),
                call().create_run(experiment_id=self.a_experiment_id),
            ]
            ml_flow_client_patch.assert_has_calls(settings_calls)

            actual_experiment_id = mlflow_logger.experiment_id
            expected_experiment_id = self.a_experiment_id
            self.assertEqual(expected_experiment_id, actual_experiment_id)

            actual_run_id = mlflow_logger.run_id
            expected_run_id = self.a_run_id
            self.assertEqual(expected_run_id, actual_run_id)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", side_effect=[a_git_commit])
    def test_whenGitRepo_givenAMLFlowInstantiation_thenLogGitCommit(self, get_git_commit_patch):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_run = self.run_mock
            MLFlowLogger(self.a_experiment_name)

            git_logging_call = [call(self.the_working_directory)]
            get_git_commit_patch.assert_has_calls(git_logging_call)

            mlflow_client_call = [call().set_tag(self.a_run_id, mlflow_default_git_commit_tag, a_git_commit)]
            ml_flow_client_patch.assert_has_calls(mlflow_client_call)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenLogMetric_givenAMLFlowCallback_thenLogMetric(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name)

            ml_flow_client_calls = []
            for key, value in self.a_log.items():
                mlflow_logger.log_metric(key, value)
                ml_flow_client_calls.append(call().log_metric(run_id=self.a_run_id, key=key, value=value, step=None))
            ml_flow_client_patch.assert_has_calls(ml_flow_client_calls)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenLogParam_givenAMLFlowCallback_thenLogParam(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name)

            ml_flow_client_calls = []
            for key, value in self.settings_in_dict.items():
                mlflow_logger.log_param(key, value)
                ml_flow_client_calls.append(call().log_param(run_id=self.a_run_id, key=key, value=value))
            ml_flow_client_patch.assert_has_calls(ml_flow_client_calls)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenLogConfigParamsASimpleDict_givenAMLFlowCallback_thenLogParams(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name)
            mlflow_logger.log_config_params(self.settings_in_dict)

            ml_flow_client_calls = []
            for key, value in self.settings_in_dict.items():
                ml_flow_client_calls.append(call().log_param(run_id=self.a_run_id, key=key, value=value))
            ml_flow_client_patch.assert_has_calls(ml_flow_client_calls)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenLogConfigParamsAConfigDict_givenAMLFlowCallback_thenLogParams(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name)
            mlflow_logger.log_config_params(self.settings_in_dict_config_no_sequence)

            ml_flow_client_calls = self._populate_calls_from_dict(self.settings_in_dict_config_no_sequence)
            ml_flow_client_patch.assert_has_calls(ml_flow_client_calls)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenLogConfigParamsAConfigDictWithSequence_givenAMLFlowCallback_thenLogParams(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name)
            mlflow_logger.log_config_params(self.settings_in_dict_config_with_sequence)

            ml_flow_client_calls = self._populate_calls_from_dict(self.settings_in_dict_config_with_sequence)
            ml_flow_client_patch.assert_has_calls(ml_flow_client_calls)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenOnTrainEndSuccess_givenAMLFlowCallback_thenLogLastEpochNumber(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name)
            mlflow_logger.set_params({"epochs": self.num_epochs})
            mlflow_logger.on_train_end(self.a_log)

            ml_flow_client_calls = [
                call().log_metric(run_id=self.a_run_id, key='last-epoch', value=self.num_epochs, step=None)
            ]
            ml_flow_client_patch.assert_has_calls(ml_flow_client_calls)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenOnTrainEndSuccess_givenAMLFlowCallback_thenHasSuccessTerminatedStatus(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name)
            mlflow_logger.set_params({"epochs": self.num_epochs})
            mlflow_logger.on_train_end(self.a_log)

            ml_flow_client_calls = [call().set_terminated(self.a_run_id, status="FINISHED")]
            ml_flow_client_patch.assert_has_calls(ml_flow_client_calls)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenOnTestFailure_givenAMLFlowCallback_thenHasFailureTerminatedStatus(self):
        # pylint: disable=protected-access
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name)
            mlflow_logger._status_handling()

            ml_flow_client_calls = [call().set_terminated(self.a_run_id, status="FAILED")]
            ml_flow_client_patch.assert_has_calls(ml_flow_client_calls)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_whenOnTestSuccess_givenAMLFlowCallback_thenHasSuccessTerminatedStatus(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name)
            mlflow_logger.set_params({"epochs": self.num_epochs})
            mlflow_logger.on_train_end(self.a_log)
            mlflow_logger.on_test_begin({})  # since we change status at the start of testing
            mlflow_logger.on_test_end(self.a_log)

            ml_flow_client_calls = [call().set_terminated(self.a_run_id, status="FINISHED")]
            ml_flow_client_patch.assert_has_calls(ml_flow_client_calls)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_integration_train(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name)

            torch.manual_seed(42)
            a_pytorch_network = nn.Linear(1, 1)
            a_loss_function = nn.MSELoss()
            an_optimizer = torch.optim.Adam(a_pytorch_network.parameters(), lr=1e-3)
            model = Model(a_pytorch_network, an_optimizer, a_loss_function)

            train_generator = SomeDataGeneratorWithLen(self.batch_size, 10, 0)
            valid_generator = SomeDataGeneratorWithLen(self.batch_size, 10, 0)
            logs = model.fit_generator(
                train_generator,
                valid_generator,
                epochs=self.num_epochs,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=[mlflow_logger],
            )

            ml_flow_client_calls = self._populate_calls_from_logs(logs)
            ml_flow_client_patch.assert_has_calls(ml_flow_client_calls)

    @patch("poutyne.framework.mlflow_logger._get_git_commit", MagicMock())
    def test_integration_train_batch_granularity(self):
        with patch("poutyne.framework.mlflow_logger.MlflowClient") as ml_flow_client_patch:
            ml_flow_client_patch.return_value.create_experiment = self.experiment_mock
            ml_flow_client_patch.return_value.create_run = self.run_mock

            mlflow_logger = MLFlowLogger(self.a_experiment_name, batch_granularity=True)

            torch.manual_seed(42)
            a_pytorch_network = nn.Linear(1, 1)
            a_loss_function = nn.MSELoss()
            an_optimizer = torch.optim.Adam(a_pytorch_network.parameters(), lr=1e-3)
            model = Model(a_pytorch_network, an_optimizer, a_loss_function)

            train_generator = SomeDataGeneratorWithLen(self.batch_size, 10, 0)
            valid_generator = SomeDataGeneratorWithLen(self.batch_size, 10, 0)
            a_num_epoch = 2
            model.fit_generator(
                train_generator,
                valid_generator,
                epochs=a_num_epoch,
                steps_per_epoch=self.steps_per_epoch,
                callbacks=[mlflow_logger],
            )

            self._assert_has_granularity_calls(ml_flow_client_patch)

    def _populate_calls_from_dict(self, config_dict: Dict) -> List:
        ml_flow_client_calls = []
        for key, value in config_dict.items():
            if isinstance(value, Mapping):
                for key_lower, value_lower in value.items():
                    good_key = f"{key}.{key_lower}"
                    ml_flow_client_calls.append(call().log_param(run_id=self.a_run_id, key=good_key, value=value_lower))
            elif isinstance(value, Sequence) and not isinstance(value, str):
                for idx, value_lower in enumerate(value):
                    good_key = f"{key}.{idx}"
                    ml_flow_client_calls.append(
                        (call().log_param(run_id=self.a_run_id, key=good_key, value=value_lower))
                    )
            else:
                ml_flow_client_calls.append(call().log_param(run_id=self.a_run_id, key=key, value=value))
        return ml_flow_client_calls

    def _populate_calls_from_logs(self, logs: Dict) -> List:
        ml_flow_client_calls = []
        for epoch_num, epoch_log in enumerate(logs):
            epoch_log.pop("epoch")
            for key, value in epoch_log.items():
                ml_flow_client_calls.append(
                    call().log_metric(run_id=self.a_run_id, key=key, value=value, step=epoch_num + 1)
                )  # +1 for enumerate
        ml_flow_client_calls.append(
            call().log_metric(run_id=self.a_run_id, key="last-epoch", value=self.num_epochs, step=None)
        )
        ml_flow_client_calls.append(call().set_terminated(self.a_run_id, status='FINISHED'))
        return ml_flow_client_calls

    def _assert_has_granularity_calls(self, ml_flow_client_patch):
        for _ in range(1, self.num_epochs):
            for step_number in range(1, self.steps_per_epoch):
                ml_flow_client_step_calls = []
                ml_flow_client_step_calls.append(
                    call().log_metric(run_id=self.a_run_id, key='batch', value=step_number, step=step_number)
                )
                ml_flow_client_step_calls.append(
                    call().log_metric(run_id=self.a_run_id, key='size', value=self.batch_size, step=step_number)
                )
                ml_flow_client_patch.assert_has_calls(ml_flow_client_step_calls)


class GetGitCommitTest(TestCase):
    def setUp(self) -> None:
        self.a_fake_path = "a_fake_path"
        self.a_wrong_path = "/a_wrong_path"
        self.a_git_sha = a_git_commit

    @patch("poutyne.framework.mlflow_logger.git", None)
    def test_whenGitNotInstall_givenARepositoryPathToGetGitCommit_thenRaiseWarning(self):
        with self.assertWarns(UserWarning):
            _get_git_commit(self.a_fake_path)

    @patch("poutyne.framework.mlflow_logger.git", None)
    def test_whenGitNotInstall_givenARepositoryPathToGetGitCommit_thenGitCommitIsNone(self):
        with self.assertWarns(UserWarning):
            commit = _get_git_commit(self.a_fake_path)
        self.assertIsNone(commit)

    @patch("poutyne.framework.mlflow_logger.git.Repo")
    def test_whenGitInstalled_givenARepositoryPathToGetGitCommitButNotAGitRepo_thenRaiseWarning(self, git_repo_patch):
        git_repo_patch.side_effect = git.NoSuchPathError()
        with self.assertWarns(UserWarning):
            _get_git_commit(self.a_fake_path)

    def test_whenGitInstalled_givenAWrongRepositoryPathToGetGitCommit_thenRaiseWarning(self):
        with self.assertWarns(UserWarning):
            _get_git_commit(self.a_wrong_path)

    @patch("poutyne.framework.mlflow_logger.git.Repo")
    def test_whenGitInstalled_givenARepositoryPathToGetGitCommitAndAGitRepo_thenReturnCommit(self, git_repo_patch):
        git_repo_patch.return_value.head.commit.hexsha = self.a_git_sha

        actual = _get_git_commit(self.a_fake_path)
        expected = self.a_git_sha

        self.assertEqual(expected, actual)
