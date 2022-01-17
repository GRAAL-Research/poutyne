# pylint: disable=line-too-long, pointless-string-statement
import os
import warnings
from typing import Dict, Optional, List

import torch

from . import Logger

try:
    import wandb

except ImportError:
    wandb = None


class WandBLogger(Logger):
    """

    WandB logger to manage logging of experiments parameters, metrics update, models log, gradient values and other information. The
    logger will log all run into the same experiment.

    Args:
        name(str): Display name for the run.
        group (Optional[str]): the name of the group to which this run belongs.
        config (Optional[Dict]): a dictionary summarizing the configuration
                related to the current run.
        save_dir(str): Path where data is saved (wandb dir by default).
        offline(bool): Run offline (data can be streamed later to wandb servers).
        id(str): Sets the version, mainly used to resume a previous run.
        version(str): Same as id.
        anonymous(bool): Enables or explicitly disables anonymous logging.
        project(str): The name of the project to which this run will belong.
        experiment: Experiment to use instead of creating a new one.
        batch_granularity(bool): Whether to also output the result of each batch in addition to the epochs.
            (Default value = False).
        checkpoints_path (Optional[str]): a string leading to the checkpoint saving directory.
                Specify this argument if you which to log the model checkpoints at the end of the
                training phase.
        initial_artifacts_paths (Optional[List[str]]): a list of paths leading to artifacts
                to be logged before the start of the training.

        log_gradient_frequency(int): log gradients and parameters every N batches (Default value = None).
        training_batch_shape(tuples): Shape of a training batch. It will be used for logging architecture of the model on wandb


    Example:
        .. code-block:: python

            wandb_logger = WandBLogger(name="First_run",project="Test_project" save_dir="/absolute/path/to/directory", experiment="First experiment")
            wandb_logger.log_config_params(config_params=cfg_dict) # logging the config dictionary

            # our Poutyne experiment
            experiment = Experiment(directory=saving_directory, network=network, device=device, optimizer=optimizer,
                            loss_function=cross_entropy_loss, batch_metrics=[accuracy])

            # Using the WandB logger callback during training
            experiment.train(train_generator=train_loader, valid_generator=valid_loader, epochs=1,
                             seed=42, callbacks=[wandb_logger])

            # You can access the wandb run via the attribute .run if you want to use other wandb features
            image = wandb.Image(an_image, caption="a caption")
            wandb_logger.run.log({"an example": image})

    """

    def __init__(
            self,
            name: Optional[str] = None,
            group: Optional[str] = None,
            config: Optional[Dict] = None,
            save_dir: Optional[str] = None,
            offline: Optional[bool] = False,
            id: Optional[str] = None,
            anonymous: Optional[bool] = None,
            version: Optional[str] = None,
            project: Optional[str] = None,
            experiment=None,
            batch_granularity: Optional[bool] = False,
            checkpoints_path: Optional[str] = None,
            initial_artifacts_paths: Optional[List[str]] = None,
            log_gradient_frequency: Optional[int] = None,
            training_batch_shape: Optional[tuple] = None,
    ) -> None:

        super().__init__(batch_granularity=batch_granularity)

        if wandb is None:
            raise ImportError("WandB needs to be installed to use this callback.")

        anonymous_lut = {True: "allow", False: None}
        self._wandb_init = dict(
            name=name,
            group=group,
            config=config,
            project=project,
            id=version or id,
            dir=save_dir,
            resume="allow",
            anonymous=anonymous_lut.get(anonymous, anonymous),
        )

        if experiment is None:

            if offline:
                os.environ["WANDB_MODE"] = "dryrun"

            if wandb.run is None:
                self.run = wandb.init(**self._wandb_init)
            else:
                warnings.warn(
                    "There is already a wandb run experience running. This callback will reuse this run. If you want "
                    "to start a new one stop this process and call `wandb.finish()` before starting again."
                )
                self.run = wandb.run
        else:
            self.run = experiment

        self.run.config.update({"run_id": self.run.id})
        self.log_gradient_frequency = log_gradient_frequency
        self.training_batch_shape = training_batch_shape
        self.checkpoints_path = checkpoints_path

    def _watch_gradient(self) -> None:
        """
        Activate wandb gradient watching.
        """
        self.run.watch(self.model.network, log="all", log_freq=self.log_gradient_frequency)

    def _save_architecture(self) -> None:
        """
        Save architecture.
        """
        dummies_batch = torch.randn(self.training_batch_shape)
        save_path = self.run.dir + "/" + self.run.name + "_model.onnx"
        torch.onnx.export(self.model.network, dummies_batch, save_path)
        self.run.save(save_path)

    def on_train_begin(self, logs: Dict):
        super().on_train_begin(logs)
        if self.log_gradient_frequency is not None:
            self._watch_gradient()

        if self.training_batch_shape is not None:
            self._save_architecture()

    def log_config_params(self, config_params: Dict) -> None:
        """
        Args:
            config_params Dict:
                Dictionary of config parameters of the training to log, such as number of epoch, loss function, 
                optimizer etc.
        """
        self.run.config.update(config_params)

    def _on_train_batch_end_write(self, batch_number: int, logs: Dict) -> None:
        """
        Log the batch metric.
        """
        if self.batch_granularity:
            train_metrics = {key: value for (key, value) in logs.items() if "val_" not in key}
            train_metrics = {"training": {"batch": train_metrics}}
            self._log_metrics(train_metrics, step=batch_number)

    def _on_epoch_end_write(self, epoch_number: int, logs: Dict) -> None:
        """
        Log the epoch metric.
        """
        train_metrics = {key: value for (key, value) in logs.items() if "val_" not in key}
        val_metrics = {key.replace("val_", ""): value for (key, value) in logs.items() if "val_" in key}
        learning_rate = self._get_current_learning_rates()

        if self.batch_granularity:
            train_metrics = {"training": {"epoch": train_metrics}}
        else:
            train_metrics = {"training": train_metrics}

        val_metrics = {"validation": val_metrics}

        self._log_metrics(train_metrics, step=epoch_number)
        self._log_metrics(val_metrics, step=epoch_number)
        self._log_params(learning_rate, step=epoch_number)

    def _on_train_end_write(self, logs: Dict):
        if self.checkpoints_path is not None:
            self._log_artifacts([self.checkpoints_path], "Checkpoints", artifact_type="Model-weights")

    def _log_metrics(self, metrics: Dict, step: int):
        """
        Log metrics for a specific step.

        Args:
            metrics (Dict): the metrics to log in the form of a dictionary.
            step (int): the corresponding step.
        """
        self.run.log(metrics)  # , step=step)

    def _log_params(self, params: Dict, step: int):
        """
        Log parameters for a specific step.
        This functions logs parameters as metrics since wandb doesn't support
        parameter logging. However, the logged parameters are prepended by the keyword
        `parameter` so as to easily identify them.

        Args:
            params (Dict): the parameters to log in the form of a dictionary.
            step (int): the corresponding step.
        """
        self.run.log({"params": params})  # , step=step)

    def _log_artifacts(self, paths: List[str], name: str, artifact_type: str):
        """
        Log artifacts for a specific step.
        This function logs multiple artifacts under the same artifact group. if
        you wish to log multiple artifacts alone (i.e: under different artifact
        groups), you should make multiple calls to this function.

        Args:
            paths (List[str]): a list of paths leading to the directories or files
                that are to be logged.
            name (str): the name of the artifact group.
            artifact_type (str): the type of the artifact group.
        """

        artifact = wandb.Artifact(name=name, type=artifact_type)
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The path {path} is not a file nor a directory")

            if os.path.isdir(path):
                artifact.add_dir(path)
            elif os.path.isfile(path):
                artifact.add_file(path)

        self.run.log_artifact(artifact)

    def _on_test_end_write(self, logs: Dict):
        # The test metrics are logged a step further than the training's
        # last step
        logs = {"testing": {key.replace("test_", ""): value for (key, value) in logs.items()}}
        self._log_metrics(logs, step=self.run.step + 1)

    def on_test_end(self, logs: Dict):
        self._on_test_end_write(logs)
        wandb.finish()
