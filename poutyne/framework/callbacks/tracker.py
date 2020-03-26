from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .callbacks import Callback


class WeightsGradientsStatsTracker:
    """
    The weights' gradient statistic tracker will compute the running absolute mean, running absolute mean variance, min
    and max per layer. The tracker is using the `Welford's online algorithm
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>`_ to compute the
    running absolute mean and running variance of the weights' gradients.
    """

    def __init__(self, number_layers) -> None:
        self.number_layers = number_layers

        self.reset()

    def batch_statistic_upgrade(self, named_parameters: Tuple[str, torch.nn.parameter.Parameter]) -> None:
        """
        Accumulate the running mean, running variance, min and the max for all the layers.
        Args:
             named_parameters (Tuple[str, ~torch.nn.parameter.Parameter): The named parameters of the model to
             track.
        """
        batch_layer_means = []
        batch_layer_min = []
        batch_layer_max = []

        for _, layer_params in named_parameters:
            layer_gradient = layer_params.grad

            abs_value_layer_gradient = layer_gradient.abs()

            batch_layer_means.append(abs_value_layer_gradient.mean().cpu().detach().numpy())
            batch_layer_min.append(layer_gradient.min().cpu().detach().numpy())
            batch_layer_max.append(layer_gradient.max().cpu().detach().numpy())

        batch_layer_means = np.array(batch_layer_means)
        previous_mean = self.running_mean

        self.running_mean = previous_mean + (batch_layer_means - previous_mean) / self.counter

        self.running_m2 = self.running_m2 + (batch_layer_means - previous_mean) * (batch_layer_means -
                                                                                   self.running_mean)

        self.running_variance = self.running_m2 / (self.counter - 1) if self.counter > 1 else self.running_variance

        batch_layer_min = np.array(batch_layer_min)
        batch_layer_max = np.array(batch_layer_max)

        self.running_min = np.minimum(batch_layer_min, self.running_min)
        self.running_max = np.maximum(batch_layer_max, self.running_max)

        self.counter += 1

    def get_stats(self, layer_names: List[str]) -> Dict:
        """
        Get the accumulated statistics of the layers.
        Note: This will reset the gradient tracker statistics values.
        Args:
            layer_names (List[str]): The names of the layer to get statistics from.
        Returns:
            A dictionary where the keys are the layer names and the values are the statistics of the layer.
            The statistics is also a dictionary where the keys are the logged statistics
            (mean, mean +/- std deviation, min and max) and the values are the corresponding statistic values.
        """
        formatted_stats = {}
        for index, layer_name in enumerate(layer_names):
            stats = {
                "mean": self.running_mean[index],
                "mean_std_dev_up": self.running_mean[index] + np.sqrt(self.running_variance[index]),
                "mean_std_dev_down": self.running_mean[index] - np.sqrt(self.running_variance[index]),
                "min": self.running_min[index],
                "max": self.running_max[index]
            }

            formatted_stats.update({layer_name: stats})

        self.reset()
        return formatted_stats

    def reset(self) -> None:
        """
        Reset the running mean, variance, min, max and counter values.
        """
        self.running_mean = np.zeros([self.number_layers], dtype="float32")
        self.running_variance = np.zeros([self.number_layers], dtype="float32")
        self.running_m2 = np.zeros([self.number_layers], dtype="float32")
        self.running_min = np.zeros([self.number_layers], dtype="float32")
        self.running_max = np.zeros([self.number_layers], dtype="float32")
        self.counter = 1


class Tracker(Callback):
    def __init__(self, keep_bias: bool = False) -> None:
        super().__init__()

        self.keep_bias = keep_bias
        self.writer = None

        self.layer_names = []
        self.number_layers = 0

        self.tracker = None

    def on_train_batch_end(self, batch: int, logs: Dict):
        # pylint: disable=unused-argument
        named_parameters = ((n, p) for n, p in self.model.network.named_parameters() if self._keep_layer(p, n))
        self.tracker.batch_statistic_upgrade(named_parameters)

    def on_train_begin(self, logs: Dict):
        for layer_name, layer_params in self.model.network.named_parameters():
            self._update_layers_to_track(layer_name, layer_params)
        self.tracker = WeightsGradientsStatsTracker(self.number_layers)

    def on_train_end(self, logs: Dict):
        self.writer.close()

    def on_epoch_end(self, epoch: int, logs: Dict):
        self._on_epoch_end_log(epoch, logs)

    def _on_epoch_end_log(self, epoch: int, logs: Dict):
        """
        The method to define the behavior of the logging tracker into a writer.

        Args:
            epoch (int): The epoch number.
            logs (Dict): The epoch logs dictionary.
        """
        pass

    def _update_layers_to_track(self, layer_name: str, layer_params: torch.nn.parameter.Parameter):
        if self._keep_layer(layer_params, layer_name):
            self.layer_names.append(layer_name)

        self.number_layers = len(self.layer_names)

    def _keep_layer(self, layer_params: torch.nn.parameter.Parameter, layer_name: str):
        layer_require_grad = layer_params.requires_grad
        if self.keep_bias:
            return layer_require_grad
        return layer_require_grad and ("bias" not in layer_name)


class TensorBoardGradientTracker(Tracker):
    """
    Wrapper to track the statistics of the weights' gradient per layer and log them in TensorBoard per epoch.
    args:
        writer (~torch.utils.tensorboard.writer.SummaryWriter): The TensorBoard writer.
        keep_bias (bool): Either or not to log the bias of the network.
    """

    def __init__(self, writer: SummaryWriter, keep_bias: bool = False) -> None:
        super().__init__(keep_bias)

        self.writer = writer

    def _on_epoch_end_log(self, epoch: int, logs: Dict) -> None:
        gradient_distributions_stats = ["mean", "mean_std_dev_up", "mean_std_dev_down"]
        other_gradient_stats = ["min", "max"]

        formatted_stats = self.tracker.get_stats(self.layer_names)
        for layer_name in self.layer_names:
            stats = formatted_stats[layer_name]

            for gradient_distributions_stat in gradient_distributions_stats:
                self.writer.add_scalars('gradient_distributions/{}'.format(layer_name),
                                        {gradient_distributions_stat: stats[gradient_distributions_stat]}, epoch)
            for other_gradient_stat in other_gradient_stats:
                self.writer.add_scalars('other_gradient_stats/{}'.format(layer_name),
                                        {other_gradient_stat: stats[other_gradient_stat]}, epoch)
