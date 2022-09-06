import os
import warnings
from tempfile import TemporaryDirectory, TemporaryFile

from unittest.mock import patch, MagicMock, call
from unittest import TestCase, main


import torch
import torch.nn as nn
import wandb

from tests.framework.tools import some_data_generator
from poutyne import Model, WandBLogger, ModelCheckpoint, Callback


class History(Callback):
    def on_epoch_end(self, epoch_number, logs):
        is_epoch = True
        self.history.append((is_epoch, logs))

    def on_train_batch_end(self, batch_number, logs):
        is_epoch = False
        self.history.append((is_epoch, logs))

    def on_train_begin(self, logs):
        self.history = []


class WandBLoggerTest(TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.a_lr = 1e-3
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=self.a_lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.num_epochs = 2
        self.a_name = "test_run"
        self.a_dir = os.getcwd()
        self.anonymous_lut = {True: "allow", False: None}
        self.run_mock = MagicMock(spec=wandb.sdk.wandb_run.Run)
        self.artifact_mock = MagicMock(spec=wandb.sdk.wandb_artifacts.Artifact)
        self.initialize_experiment = MagicMock(return_value=self.run_mock)
        self.a_config_params = {"param_1": 1, "param_2": 2, "param_3": "value"}
        self.uncleaned_log = {"size": 32, "accuracy": 85}
        self.temp_dir_obj = TemporaryDirectory()
        self.temp_file_obj = TemporaryFile()

    @patch.dict(os.environ, clear=True)
    def test_wandb_online_init(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            wandb_patch.run = None
            WandBLogger(name=self.a_name)
            create_experiment_call = [
                call.init(
                    name=self.a_name,
                    group=None,
                    config=None,
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
            WandBLogger(name=self.a_name, offline=True)
            create_experiment_call = [
                call.init(
                    name=self.a_name,
                    group=None,
                    config=None,
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
                WandBLogger(name=self.a_name)
            self.assertEqual(len(w), 1)
            wandb_patch.assert_not_called()

    def test_with_run_init(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            wandb_patch.run = self.run_mock
            wandb_logger = WandBLogger(name=self.a_name, experiment=self.run_mock)
            wandb_patch.assert_not_called()
            self.assertIsInstance(wandb_logger.run, wandb.sdk.wandb_run.Run)

    def test_log_config(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            wandb_patch.init = self.initialize_experiment
            wandb_patch.run = None
            logger = WandBLogger(name=self.a_name)
            logger.log_config_params(self.a_config_params)

            create_experiment_call = [call(self.a_config_params)]
            logger.run.config.update.assert_has_calls(create_experiment_call)

    def test_wandb_with_artifact_file_init(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            wandb_patch.Artifact.return_value = self.artifact_mock
            wandb_patch.init = self.initialize_experiment
            wandb_patch.run = None
            logger = WandBLogger(name=self.a_name, initial_artifacts_paths=[self.temp_file_obj.name])
            self.artifact_mock.add_file.assert_called_once_with(self.temp_file_obj.name)
            logger.run.log_artifact.assert_called_once_with(self.artifact_mock)

    def test_wandb_with_artifact_dir_init(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            wandb_patch.Artifact.return_value = self.artifact_mock
            wandb_patch.init = self.initialize_experiment
            wandb_patch.run = None
            logger = WandBLogger(name=self.a_name, initial_artifacts_paths=[self.temp_dir_obj.name])
            self.artifact_mock.add_dir.assert_called_once_with(self.temp_dir_obj.name)
            logger.run.log_artifact.assert_called_once_with(self.artifact_mock)

    def test_watch_gradient_false(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            train_gen = some_data_generator(20)
            valid_gen = some_data_generator(20)
            wandb_patch.init = self.initialize_experiment
            wandb_patch.run = None
            logger = WandBLogger(name=self.a_name)

            self.model.fit_generator(
                train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
            )

            logger.run.watch.assert_not_called()

    def test_watch_gradient_true(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            train_gen = some_data_generator(20)
            valid_gen = some_data_generator(20)
            wandb_patch.init = self.initialize_experiment
            wandb_patch.run = None
            logger = WandBLogger(name=self.a_name, log_gradient_frequency=1)
            self.model.fit_generator(
                train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
            )

            logger.run.watch.assert_called_once_with(self.pytorch_network, log="all", log_freq=1)

    def test_log_epoch(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            train_gen = some_data_generator(20)
            valid_gen = some_data_generator(20)
            wandb_patch.init = self.initialize_experiment
            wandb_patch.run = None
            logger = WandBLogger(name=self.a_name, log_gradient_frequency=1, batch_granularity=False)
            history = self.model.fit_generator(
                train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger]
            )
            experiment_call = []
            for log in history:
                epoch = log["epoch"]
                train_metrics = {key: value for (key, value) in log.items() if not key.startswith("val_")}
                train_metrics = {"training": train_metrics}
                experiment_call.append(call.log(train_metrics, step=epoch))

                val_metrics = {key[4:]: value for (key, value) in log.items() if key.startswith("val_")}
                val_metrics = {"validation": val_metrics}
                experiment_call.append(call(val_metrics, step=epoch))

                experiment_call.append(call({"params": {"lr": self.a_lr}}, step=epoch))

            logger.run.log.assert_has_calls(experiment_call, any_order=False)

    def test_log_epoch_and_batch(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            train_gen = some_data_generator(20)
            valid_gen = some_data_generator(20)
            wandb_patch.init = self.initialize_experiment
            wandb_patch.run = None
            num_batchs = 5
            logger = WandBLogger(name=self.a_name, log_gradient_frequency=1, batch_granularity=True)
            history = History()
            self.model.fit_generator(
                train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=num_batchs, callbacks=[logger, history]
            )
            experiment_call = []
            for is_epoch, log in history.history:
                if is_epoch:
                    train_metrics = {key: value for (key, value) in log.items() if not key.startswith("val_")}
                    train_metrics = {"training": {"epoch": train_metrics}}

                    experiment_call.append(call.log(train_metrics))
                    val_metrics = {key[4:]: value for (key, value) in log.items() if key.startswith("val_")}
                    val_metrics = {"validation": {"epoch": val_metrics}}

                    experiment_call.append(call(val_metrics))

                    experiment_call.append(call({"params": {"lr": self.a_lr}}))
                else:
                    train_metrics = {key: value for (key, value) in log.items() if not key.startswith("val_")}
                    train_metrics = {"training": {"batch": train_metrics}}
                    experiment_call.append(call(train_metrics))

            logger.run.log.assert_has_calls(experiment_call, any_order=False)

    def test_log_testgenerator(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            x = torch.rand(10, 1)
            y = torch.rand(10, 1)
            wandb_patch.init = self.initialize_experiment
            wandb_patch.run = None
            logger = WandBLogger(name=self.a_name, log_gradient_frequency=1, batch_granularity=True)
            self.model.evaluate(x, y, callbacks=[logger])
            logger.run.log.called_once()

    def test_log_checkpoints(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            train_gen = some_data_generator(20)
            valid_gen = some_data_generator(20)
            wandb_patch.init = self.initialize_experiment
            wandb_patch.run = None
            wandb_patch.Artifact.return_value = self.artifact_mock
            tmp_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint.tmp.ckpt')
            checkpoint_filename = os.path.join(self.temp_dir_obj.name, 'my_checkpoint.ckpt')
            checkpointer = ModelCheckpoint(
                checkpoint_filename, monitor='val_loss', verbose=True, period=1, temporary_filename=tmp_filename
            )
            logger = WandBLogger(
                name=self.a_name,
                log_gradient_frequency=1,
                batch_granularity=True,
                checkpoints_path=self.temp_dir_obj.name,
            )
            self.model.fit_generator(
                train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger, checkpointer]
            )
            self.artifact_mock.add_dir.assert_called_once_with(self.temp_dir_obj.name)
            logger.run.log_artifact.assert_called_once_with(self.artifact_mock)

    def test_save_architecture(self):
        with patch("poutyne.framework.wandb_logger.wandb") as wandb_patch:
            with patch("poutyne.framework.torch.onnx") as torch_onx_patch:
                with patch("poutyne.framework.torch.randn") as torch_randn_patch:
                    train_gen = some_data_generator(20)
                    valid_gen = some_data_generator(20)

                    wandb_patch.init = self.initialize_experiment
                    wandb_patch.run = None
                    num_batchs = 5
                    logger = WandBLogger(
                        name=self.a_name,
                        log_gradient_frequency=1,
                        batch_granularity=True,
                        training_batch_shape=(1, 2, 3),
                    )
                    logger.run.dir = "a_path"
                    logger.run.name = self.a_name
                    self.model.fit_generator(
                        train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=num_batchs, callbacks=[logger]
                    )

                    torch_onx_patch.export.assert_called_once_with(
                        self.pytorch_network, torch_randn_patch().to(), f"a_path/{self.a_name}_model.onnx"
                    )
                    logger.run.save.assert_called_once_with(f"a_path/{self.a_name}_model.onnx")


if __name__ == '__main__':
    main()
