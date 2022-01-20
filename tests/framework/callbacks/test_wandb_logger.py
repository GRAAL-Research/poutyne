import unittest
from unittest import TestCase, main

import torch
import torch.nn as nn
import os
import warnings
import wandb

from poutyne import Model, WandBLogger
from tests.framework.tools import some_data_generator

from unittest.mock import patch, MagicMock, call

# Ajouter batch granuality, tester history

class WandBLoggerTest(TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.num_epochs = 2
        self.a_name = "test_run"
        self.a_dir = os.getcwd()
        self.anonymous_lut = {True: "allow", False: None}
        self.run_mock = MagicMock(spec=wandb.sdk.wandb_run.Run)
        self.initialize_experience = MagicMock(return_value=self.run_mock)
        self.a_config_params = {"param_1": 1, "param_2": 2, "param_3": "value"}
        self.uncleaned_log = {"size": 32, "accuracy": 85}

    @patch.dict(os.environ, clear=True)
    def test_wandb_online_init(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            wandb_patch.run = None
            WandBLogger(self.a_name)
            create_experiment_call = [
                call.init(
                    name=self.a_name,
                    project=None,
                    id=None,
                    dir=None,
                    resume="allow",
                    anonymous=self.anonymous_lut.get(None, None),
                )
            ]
            self.assertIsNone(os.getenv("WANDB_MODE"))
            wandb_patch.assert_has_calls(create_experiment_call)

    @patch.dict(os.environ, {"WANDB_MODE": "dryrun"}, clear=True)
    def test_wandb_offline_init(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            wandb_patch.run = None
            WandBLogger(self.a_name, offline=True)
            create_experiment_call = [
                call.init(
                    name=self.a_name,
                    project=None,
                    id=None,
                    dir=None,
                    resume="allow",
                    anonymous=self.anonymous_lut.get(None, None),
                )
            ]
            self.assertEqual(os.getenv("WANDB_MODE"), "dryrun")
            wandb_patch.assert_has_calls(create_experiment_call)



    def test_already_running_warning_init(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            wandb_patch.run = self.run_mock
            with warnings.catch_warnings(record=True) as w:
                WandBLogger(self.a_name)
            self.assertEqual(len(w), 1)
            wandb_patch.assert_not_called()

    def test_with_run_init(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            wandb_patch.run = self.run_mock
            wandb_logger = WandBLogger(self.a_name, experiment=self.run_mock)
            wandb_patch.assert_not_called()
            self.assertIsInstance(wandb_logger.run, wandb.sdk.wandb_run.Run)

    def test_log_config(self):

        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            wandb_patch.init = self.initialize_experience
            wandb_patch.run = None
            wandb_logger = WandBLogger(self.a_name)
            wandb_logger.log_config_params(self.a_config_params)

            create_experiment_call = [call(self.a_config_params)]
            self.run_mock.config.update.assert_has_calls(create_experiment_call)

    def test_watch_gradient_false(self):

        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            train_gen = some_data_generator(20)
            valid_gen = some_data_generator(20)
            wandb_patch.init = self.initialize_experience
            wandb_patch.run = None
            logger = WandBLogger(self.a_name)
            history = self.model.fit_generator(
                train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
            )

            self.run_mock.watch.assert_not_called()

    def test_watch_gradient_true(self):

        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            train_gen = some_data_generator(20)
            valid_gen = some_data_generator(20)
            wandb_patch.init = self.initialize_experience
            wandb_patch.run = None
            logger = WandBLogger(self.a_name, log_gradient_frequency=1)
            history = self.model.fit_generator(
                train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
            )

            self.run_mock.watch.assert_called_once_with(self.pytorch_network, log="all", log_freq=1)
            
    def test_watch_gradient_true(self):

        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            train_gen = some_data_generator(20)
            valid_gen = some_data_generator(20)
            wandb_patch.init = self.initialize_experience
            wandb_patch.run = None
            logger = WandBLogger(self.a_name, log_gradient_frequency=1)
            history = self.model.fit_generator(
                train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
            )

            self.run_mock.watch.assert_called_once_with(self.pytorch_network, log="all", log_freq=1)

    def test_log_epoch(self):

        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            train_gen = some_data_generator(20)
            valid_gen = some_data_generator(20)
            wandb_patch.init = self.initialize_experience
            wandb_patch.run = None
            logger = WandBLogger(self.a_name, log_gradient_frequency=1)
            history = self.model.fit_generator(
                train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
            )
            experience_call = [call.log(log) for log in history]

            self.run_mock.log.assert_has_calls(experience_call,any_order=False)



if __name__ == '__main__':
    main()
