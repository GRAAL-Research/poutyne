# pylint: disable=line-too-long, pointless-string-statement
import os
import warnings
from typing import Dict, Union, Mapping, Sequence

from . import Logger

try:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None

try:
    import git
except ImportError:
    git = None

warnings.filterwarnings("ignore")


class MLFlowLogger(Logger):
    """
    MLflow logger to manage logging of experiments parameters, metrics update, models log and other information. The
    logger will log all run into the same experiment.

    Args:
        experiment_name (str): The name of the experiment. Name must be unique and are case sensitive.
        tracking_uri (Union[str, None]): Either the URI tracking path (for server tracking) of the absolute path to
            the directory to save the files (for file store). For example: http://<ip address>:<port> (remote server) or
            /home/<user>/mlflow-server (local server). If None, will use the default MLflow file
            tracking URI "./mlruns".
        batch_granularity (bool): Whether to also output the result of each batch in addition to the epochs.
            (Default value = False)

    Example:
        Using file store::

            mlflow_logger = MLFlowLogger(experiment_name="experiment", tracking_uri="/absolute/path/to/directory")
            mlflow_logger.log_config_params(config_params=cfg_dict) # logging the config dictionary

            # our Poutyne experiment
            experiment = Experiment(directory=saving_directory, network=network, device=device, optimizer=optimizer,
                            loss_function=cross_entropy_loss, batch_metrics=[accuracy])

            # Using the MLflow logger callback during training
            experiment.train(train_generator=train_loader, valid_generator=valid_loader, epochs=1,
                             seed=42, callbacks=[mlflow_logger])

        Using server tracking::

            mlflow_logger = MLFlowLogger(experiment_name="experiment", tracking_uri="http://IP_ADDRESS:PORT")
            mlflow_logger.log_config_params(config_params=cfg_dict) # logging the config dictionary

            # our Poutyne experiment
            experiment = Experiment(directory=saving_directory, network=network, device=device, optimizer=optimizer,
                            loss_function=cross_entropy_loss, batch_metrics=[accuracy])

            # Using the MLflow logger callback during training
            experiment.train(train_generator=train_loader, valid_generator=valid_loader, epochs=1,
                             seed=42, callbacks=[mlflow_logger])
    """

    def __init__(self,
                 experiment_name: str,
                 tracking_uri: Union[str, None] = None,
                 batch_granularity: bool = False) -> None:
        super().__init__(batch_granularity=batch_granularity)
        if mlflow is None:
            raise ImportError("Mlflow needs to be installed to use this callback.")

        self.tracking = tracking_uri

        self._working_directory = os.getcwd()  # for Git

        self.ml_flow_client = MlflowClient(tracking_uri=self.tracking)

        self._handle_experiment_id(experiment_name)
        self.run_id = self.ml_flow_client.create_run(experiment_id=self.experiment_id).info.run_id

        self._log_git_version()

        self._status = "FAILED"  # base case is a failure

    def log_config_params(self, config_params: Union[Mapping, Sequence]) -> None:
        """
        Args:
            config_params (Union[Mapping, Sequence]):
                The config parameters of the training to log, such as number of epoch, loss function, optimizer etc.
        """
        if isinstance(config_params, Dict):
            for param_name, element in config_params.items():
                self.log_param(param_name, element)
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

    def _log_config_write(self, parent_name: str, element: Union[int, float, str, Mapping, Sequence]) -> None:
        """
        Log the config parameters when it's a mapping or a sequence of elements.
        """
        if isinstance(element, Mapping):
            for key, value in element.items():
                # We recursively open the element (Dict format type)
                self._log_config_write("{}.{}".format(parent_name, key), value)
        elif isinstance(element, Sequence) and not isinstance(element, str):
            # Since str are sequence we negate it to be log in the else
            for idx, value in enumerate(element):
                self.log_param("{}.{}".format(parent_name, idx), value)
        else:
            self.log_param(parent_name, element)

    def _on_train_batch_end_write(self, batch_number: int, logs: Dict) -> None:
        """
        Log the batch metric.
        """
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
        self._status = "FINISHED"

        mlflow.end_run()
        self._status_handling()

    def _on_train_end_write(self, logs) -> None:
        """
        Log the last epoch batch and epoch metric.
        """
        last_epoch = self.params["epochs"]
        self.log_metric("last-epoch", last_epoch)

    def on_test_begin(self, logs: Dict):
        self._status = "FAILED"  # to change status from FINISHED to FAILED (base case) if trained before
        self._status_handling()

    def on_test_end(self, logs: Dict):
        """
        Log the test results.
        """
        if len(logs) > 0:  # to manage failure of the test loop
            self._on_test_end_write(logs)
            self._status = "FINISHED"

        mlflow.end_run()
        self._status_handling()

    def _on_test_end_write(self, logs: Dict) -> None:
        for key, value in logs.items():
            self.log_metric(key, value)

    def _status_handling(self):
        # we set_terminated the run to get the finishing status (FINISHED or FAILED)
        self.ml_flow_client.set_terminated(self.run_id, status=self._status)

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
        source_version = _get_git_commit(self._working_directory)
        if source_version is not None:
            self.ml_flow_client.set_tag(self.run_id, "mlflow.source.git.commit", source_version)


"""
The source code of the function _get_git_commit was inspired from the MLflow project
(https://github.com/mlflow/mlflow/blob/7fde53e497c50b4eb4da1e9988710695b8c2e093/mlflow/tracking/context/git_context.py#L11),
and has been modified. All modifications made from the original source code are under the LGPLv3 license.

COPYRIGHT

All contributions by MLflow:
Copyright (c) 2020, MLFlow.
All rights reserved.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.
"""


def _get_git_commit(path):
    """
    Function to get the git commit from a path.
    """
    if git is None:
        warnings.warn("Failed to import Git (the Git executable is probably not on your PATH),"
                      " so Git SHA is not available.")
        return None

    try:
        repo = git.Repo(path, search_parent_directories=True)
        commit = repo.head.commit.hexsha
        return commit
    except (git.InvalidGitRepositoryError, git.NoSuchPathError) as e:
        warnings.warn(f"Failed to grab the git repository so Git SHA is not available. Error: {e}")
        return None
