from typing import Dict, List, Optional
import os

from . import Logger

try:
    import wandb
except ImportError:
    wandb = None


class WandbLogger(Logger):
    def __init__(
        self,
        project_name: str,
        groupe_name: Optional[str] = None,
        config: Optional[Dict] = None,
        initial_artifacts_paths: Optional[List[str]] = None,
        run_id: Optional[str] = None,
        checkpoints_path: Optional[str] = None,
        batch_granularity: Optional[bool] = False,
    ):
        """
        Logging callback for Weights & Biases allowing to log intial  configurations,
        training and testing metrics, artifacts and evolving parameters such as learning rates.

        Args:
            project_name (str): the name of the project ubder which the current run is to be logged.
            groupe name (Optional[str]): the name of the group to which this run belongs.
            config (Optional[Dict]): a dictionnary sumerizing the configuration
                related to the current run.
            initial_artifacts_paths (Optional[List[str]]): a list of paths leading to artifacts
                to be logged before the start of the training.
            run_id (Optional[str]): the unique id of a specific run in case you which to resume it.
                The id of each run is automatically logged as part of the configuration.
            checkpoints_path (Optional[str]): a string leading to the checkpoint saving directory.
                Specify this argument if you which to log the model checkpoints at the end of the
                training phase.
        """
        # pylint: disable=no-member, too-many-arguments
        super().__init__(batch_granularity=batch_granularity)
        if wandb is None:
            raise ImportError("wandb must be installed to use this callback.")

        self.checkpoints_path = checkpoints_path

        wandb.init(project=project_name, group=groupe_name, config=config, resume="allow", id=run_id, reinit=False)

        wandb.config.update({"run_id": wandb.run.id})

        if initial_artifacts_paths is not None:
            self._log_artifacts(initial_artifacts_paths, name="Initial-artifacts", artifact_type="Miscellaneous")

    def _on_epoch_end_write(self, epoch_number: int, logs: Dict):
        logs.pop("epoch")
        logs.pop("time")

        train_metrics = {key: value for (key, value) in logs.items() if "val_" not in key}
        val_metrics = {key.replace("val_", ""): value for (key, value) in logs.items() if "val_" in key}

        train_metrics = {"training": train_metrics}
        val_metrics = {"validation": val_metrics}

        learning_rate = self._get_current_learning_rates()
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
            metrics (Dict): the metrics to log in the form of a dictionnary.
            step (int): the corresponding step.
        """
        wandb.log(metrics, step=step)

    def _log_params(self, params: Dict, step: int):
        """
        Log parameters for a specific step.
        This functions logs parameters as metrics since wandb doesn't support
        parameter logging. Howerver, the logged parameters are prepended by the keyword
        `parameter` so as to easily identify them.

        Args:
            params (Dict): the parameters to log in the form of a dictionnary.
            step (int): the corresponding step.
        """
        wandb.log({"params": params}, step=step)

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

        wandb.log_artifact(artifact)

    def on_test_end(self, logs: Dict):
        self._on_test_end_write(logs)
        wandb.finish()

    def _on_test_end_write(self, logs: Dict):
        # The test metrics are logged a step further than the training's
        #   last atep
        logs = {"testing": {key.replace("test_", ""): value for (key, value) in logs.items()}}
        self._log_metrics(logs, step=wandb.run.step + 1)
