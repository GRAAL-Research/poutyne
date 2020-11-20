# pylint: disable=line-too-long
"""
The source code of this file was inspired from the Hydra-MLflow-experiment-management
(https://github.com/ymym3412/Hydra-MLflow-experiment-management/tree/master), and has been modified. All modifications
made from the original source code are under the LGPLv3 license.

COPYRIGHT

All contributions by ymym3412:
Copyright (c) 2020, ymym3412.
All rights reserved.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

----------
The source code of the function _get_git_commit was inspired from the MLflow project
(https://github.com/mlflow/mlflow/blob/7fde53e497c50b4eb4da1e9988710695b8c2e093/mlflow/tracking/context/git_context.py#L11),
and has been modified. All modifications made from the original source code are under the LGPLv3 license.

COPYRIGHT

All contributions by MLflow:
Copyright (c) 2020, ymym3412.
All rights reserved.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.
"""
import os
import warnings
from typing import Dict, Union

import numpy as np

try:
    from mlflow import pytorch, log_params, end_run, active_run, start_run, set_tracking_uri
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except ImportError:
    raise ImportError("Mlflow needs to be installed to use this callback.")

try:
    import omegaconf as omega
except ImportError:
    omega = None

try:
    import git
except ImportError:
    git = None

from poutyne.framework import Logger

warnings.filterwarnings("ignore")


class MLFlowLogger(Logger):
    """
    MLflow logger to manage logging of experiments parameters, metrics update, models log and other information. The
    logger will log all run into the same experiment.

    Args:
        experiment_name (str): The name of the experiment. Name must be unique and are case sensitive.
        saving_directory (str): Either the URI tracking path (for server tracking) of the absolute path to the directory
            to save the files. For example: /home/<user>/mlflow-server (local directory) or ...
        batch_granularity (bool): Whether to also output the result of each batch in addition to the epochs.
            (Default value = False)

    Example:
            mlflow_logger = MLFlowLogger(experiment_name="experiment", saving_directory="/absolute/path/to/directory")
            mlflow_logger.log_config_params(config_params=cfg_dict) # logging the config dictionary

            # our Poutyne experiment
            experiment = Experiment(directory=saving_directory, network=network, device=device, optimizer=optimizer,
                            loss_function=cross_entropy_loss, batch_metrics=[accuracy])

            # Using the MLflow logger callback during training
            experiment.train(train_generator=train_loader, valid_generator=valid_loader, epochs=1,
                             seed=42, callbacks=[mlflow_logger])

            # logging the last epoch model
            mlflow_logger.log_model()
    """

    def __init__(self, experiment_name: str, saving_directory: str, batch_granularity: bool = False) -> None:
        super().__init__(batch_granularity=batch_granularity)
        self.tracking = saving_directory

        self.working_directory = os.getcwd()

        self.ml_flow_client = MlflowClient(tracking_uri=self.tracking, registry_uri=self.working_directory)

        self._handle_experiment_id(experiment_name)
        self.run_id = self.ml_flow_client.create_run(experiment_id=self.experiment_id).info.run_id

        self._log_git_version()

    def log_config_params(self, config_params: Union[Dict, omega.omegaconf.DictConfig,
                                                     omega.omegaconf.ListConfig]) -> None:
        """
        Args:
            config_params (Union[Dict, ~omegaconf.dictconfig.DictConfig, ~omegaconf.listconfig.ListConfig]):
                The config parameters of the training to log, such as number of epoch, loss function, optimizer etc.
        """
        if isinstance(config_params, Dict):
            log_params(config_params)
        else:
            for param_name, element in config_params.items():
                self._log_config_write(param_name, element)

    def log_param(self, param_name: str, value: Union[str, float]) -> None:
        """
        Log the value of a the parameter into the last run of the experiment.

        Args:
            param_name (str): The name of the parameter.
            value (Union[str, float]: The value of the parameter.

        """
        self.ml_flow_client.log_param(run_id=self.run_id, key=param_name, value=value)

    def log_metric(self, metric_name: str, value: float, step: Union[int, None] = None) -> None:
        """
        Log the value of a metric into the last run of the experiment.

        Args:
            metric_name (str): The name of the metric.
            value (float): The value of the metric.
            step (Union[int, None]): The step when the metric was compute (Default = None).
        """
        self.ml_flow_client.log_metric(run_id=self.run_id, key=metric_name, value=value, step=step)

    def _log_config_write(self, parent_name: str, element: Union[Dict, omega.omegaconf.DictConfig,
                                                                 omega.omegaconf.ListConfig]) -> None:
        """
        Log the config parameters when it's a list of dictionary or a dictionary of dictionary.
        """
        if omega is None:
            raise ImportError("Omegaconf needs to be installed to log this type of dictionary.")
        if isinstance(element, omega.omegaconf.DictConfig):
            for key, value in element.items():
                if isinstance(value, (omega.omegaconf.DictConfig, omega.omegaconf.ListConfig)):
                    self._log_config_write("{}.{}".format(parent_name, key), value)
                else:
                    self.log_param("{}.{}".format(parent_name, key), value)
        elif isinstance(element, omega.omegaconf.ListConfig):
            for idx, value in enumerate(element):
                self.log_param("{}.{}".format(parent_name, idx), value)

    def _on_train_batch_end_write(self, batch_number: int, logs: Dict) -> None:
        """
        Log the batch metric.
        """
        if self.batch_granularity:
            for key, value in logs.items():
                self.log_metric(key, value, step=batch_number)

    def _on_epoch_end_write(self, epoch_number: int, logs: Dict) -> None:
        """
        Log the batch and epoch metric.
        """
        logs.pop("epoch")
        for key, value in logs.items():
            self.log_metric(key, value, step=epoch_number)

    def on_train_end(self, logs: Dict):
        """
        Log the last epoch batch and epoch metric and close the active run.
        """
        self._on_train_end_write(logs)
        end_run()

    def _on_train_end_write(self, logs) -> None:
        """
        Log the last epoch batch and epoch metric.
        """
        last_epoch = self.params["epochs"]
        self.log_metric("last-epoch", last_epoch)

    def on_test_end(self, logs: Dict):
        """
        Log the test results.
        """
        test_dict = self.format_test_logs(logs)
        self._on_test_end_write(test_dict)
        end_run()

    def _on_test_end_write(self, logs: Dict) -> None:
        for key, value in logs.items():
            self.log_metric(key, value)

    def log_model(self):
        """
        Log the trained model.
        """
        if self.model is None:
            raise AttributeError("The model is not loaded into the logger.")
        set_tracking_uri(self.tracking)  # we need to set the tracking uri to be able start the close run
        start_run(run_id=self.run_id)  # reopen run since logging of model don't use the MLflowClient
        with active_run():  # log_model use the a context manager
            pytorch.log_model(self.model.network, "trained-model")
        end_run()

    def _handle_experiment_id(self, experiment_name):
        """
        Handle if the experiment name already exist to grad the id and append new experiment to it.
        """
        try:
            self.experiment_id = self.ml_flow_client.create_experiment(experiment_name, self.tracking)
        except MlflowException:
            self.experiment_id = self.ml_flow_client.get_experiment_by_name(experiment_name).experiment_id

    def _log_git_version(self):
        """
        Log the git commit of the run.
        """
        source_version = _get_git_commit(self.working_directory)
        if source_version is not None:
            self.ml_flow_client.set_tag(self.run_id, "mlflow.source.git.commit", source_version)

    def format_test_logs(self, logs: Dict) -> Dict:
        """
        Format the test logs using the metric of the model.
        """
        test_loss, test_metrics = logs
        test_metrics_names = ['test_loss'] + \
                             ['test_' + metric_name for metric_name in self.model.metrics_names]
        test_metrics_values = np.concatenate(([test_loss], test_metrics))

        test_metrics_dict = dict(zip(test_metrics_names, test_metrics_values))
        return test_metrics_dict


def _get_git_commit(path):
    """
    Function to get the git commit from a path.
    """
    if git is None:
        warnings.warn("Failed to import Git (the Git executable is probably not on your PATH),"
                      " so Git SHA is not available.")
        return None

    try:
        if os.path.isfile(path):
            path = os.path.dirname(path)
        repo = git.Repo(path, search_parent_directories=True)
        commit = repo.head.commit.hexsha
        return commit
    except (git.InvalidGitRepositoryError, git.GitCommandNotFound, ValueError, git.NoSuchPathError) as e:
        warnings.warn(f"Failed to grab the git repository so Git SHA is not available. Error: {e}")
        return None
