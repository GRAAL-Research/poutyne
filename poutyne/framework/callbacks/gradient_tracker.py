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

from typing import Dict, List, Tuple, Iterable

import numpy as np
import torch

from .callbacks import Callback


class WeightsGradientsStatsTracker:
    """
    The weights' gradient statistic tracker will estimate the absolute mean (i.e. the mean of the absolute values of
    the weights' gradients), running absolute mean variance (i.e. the variance of the absolute mean), min, absolute min,
     max and absolute max per layer. The tracker is using the `Welford's online algorithm
     <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>`_
    to estimate the running mean and running variance of the absolute weights' gradients.

    Args:
        number_layers (int): The number of layers to track stats on.
    """

    def __init__(self, number_layers: int) -> None:
        self.number_layers = number_layers

        self.reset()

    def batch_statistic_upgrade(self, named_parameters: Iterable[Tuple[str, torch.nn.parameter.Parameter]]) -> None:
        """
        Accumulate the running absolute mean, running absolute mean variance, min, absolute min, max ant the absolute
        max for all the layers.

        Args:
             named_parameters (Iterable[Tuple[str, ~torch.nn.parameter.Parameter]): The named parameters of the model to
             track.
        """
        batch_layer_abs_means = []
        batch_layer_min = []
        batch_layer_abs_min = []
        batch_layer_max = []
        batch_layer_abs_max = []

        # Just in case we want to support second-order derivatives
        with torch.no_grad():
            for _, layer_params in named_parameters:
                layer_gradient = layer_params.grad

                abs_value_layer_gradient = layer_gradient.abs()

                batch_layer_abs_means.append(abs_value_layer_gradient.mean().cpu().numpy())

                batch_layer_min.append(layer_gradient.min().cpu().numpy())
                batch_layer_abs_min.append(abs_value_layer_gradient.min().cpu().numpy())

                batch_layer_max.append(layer_gradient.max().cpu().numpy())
                batch_layer_abs_max.append(abs_value_layer_gradient.max().cpu().numpy())

        batch_layer_abs_means = np.array(batch_layer_abs_means)
        previous_mean = self.running_abs_mean

        self.running_abs_mean = previous_mean + (batch_layer_abs_means - previous_mean) / self.count

        self.running_m2 = self.running_m2 + (batch_layer_abs_means - previous_mean) * (
            batch_layer_abs_means - self.running_abs_mean
        )

        self.running_abs_mean_var = self.running_m2 / (self.count - 1) if self.count > 1 else self.running_abs_mean_var

        batch_layer_min = np.array(batch_layer_min)
        batch_layer_max = np.array(batch_layer_max)

        self.running_min = np.minimum(batch_layer_min, self.running_min)
        self.running_abs_min = np.minimum(batch_layer_abs_min, self.running_abs_min)

        self.running_max = np.maximum(batch_layer_max, self.running_max)
        self.running_abs_max = np.maximum(batch_layer_abs_max, self.running_abs_max)

        self.count += 1

    def get_stats(self, layer_names: List[str]) -> Dict:
        """
        Get the accumulated statistics of the layers.

        Note: This will reset the gradient tracker statistics values.

        Args:
            layer_names (List[str]): The names of the layer to get statistics from.

        Returns:
            A dictionary where the keys are the layer names and the values are the statistics of the layer.
            The statistics is also a dictionary where the keys are the logged statistics
            (mean, mean +/- std deviation, min, absolute min, max and the absolute max) and the values are
            the corresponding statistic values.
        """
        formatted_stats = {}
        for index, layer_name in enumerate(layer_names):
            stats = {
                "mean": self.running_abs_mean[index],
                "mean_std_dev_up": self.running_abs_mean[index] + np.sqrt(self.running_abs_mean_var[index]),
                "mean_std_dev_down": self.running_abs_mean[index] - np.sqrt(self.running_abs_mean_var[index]),
                "min": self.running_min[index],
                "abs_min": self.running_abs_min[index],
                "max": self.running_max[index],
                "abs_max": self.running_abs_max[index],
            }

            formatted_stats.update({layer_name: stats})

        self.reset()
        return formatted_stats

    def reset(self) -> None:
        """
        Reset the running absolute mean, absolute mean variance, min, absolute min, max, absolute max and count values.
        """
        self.running_abs_mean = np.zeros([self.number_layers], dtype="float32")
        self.running_abs_mean_var = np.zeros([self.number_layers], dtype="float32")
        self.running_m2 = np.zeros([self.number_layers], dtype="float32")
        self.running_min = np.zeros([self.number_layers], dtype="float32")
        self.running_abs_min = np.zeros([self.number_layers], dtype="float32")
        self.running_max = np.zeros([self.number_layers], dtype="float32")
        self.running_abs_max = np.zeros([self.number_layers], dtype="float32")
        self.count = 1


class GradientTracker(Callback):
    def __init__(self, keep_bias: bool = False) -> None:
        super().__init__()

        self.keep_bias = keep_bias
        self.layer_names = []
        self.number_layers = 0

        self.tracker = None

    def on_train_batch_end(self, batch_number: int, logs: Dict) -> None:
        # pylint: disable=unused-argument
        named_parameters = ((n, p) for n, p in self.model.network.named_parameters() if self._keep_layer(p, n))
        self.tracker.batch_statistic_upgrade(named_parameters)

    def on_train_begin(self, logs: Dict) -> None:
        for layer_name, layer_params in self.model.network.named_parameters():
            self._update_layers_to_track(layer_name, layer_params)
        self.tracker = WeightsGradientsStatsTracker(self.number_layers)

    def on_epoch_end(self, epoch_number: int, logs: Dict) -> None:
        self._on_epoch_end_log(epoch_number, logs)

    def _on_epoch_end_log(self, epoch_number: int, logs: Dict) -> None:
        """
        The method to define the behavior of the logging tracker.

        Args:
            epoch_number (int): The epoch number.
            logs (Dict): The epoch logs dictionary.
        """
        pass

    def _update_layers_to_track(self, layer_name: str, layer_params: torch.nn.parameter.Parameter):
        if self._keep_layer(layer_params, layer_name):
            self.layer_names.append(layer_name)

        self.number_layers = len(self.layer_names)

    def _keep_layer(self, layer_params: torch.nn.parameter.Parameter, layer_name: str) -> bool:
        layer_require_grad = layer_params.requires_grad
        if self.keep_bias:
            return layer_require_grad
        return layer_require_grad and ("bias" not in layer_name)


class TensorBoardGradientTracker(GradientTracker):
    """
    Wrapper to track the statistics of the weights' gradient per layer and log them in TensorBoard per epoch.

    Args:
        writer (~torch.utils.tensorboard.writer.SummaryWriter): The TensorBoard writer.
        keep_bias (bool): Either or not to log the bias of the network.

    Example:
        Using TensorBoardGradientTracker::

            from torch.utils.tensorboard import SummaryWriter
            from poutyne import Model, TensorBoardGradientTracker

            writer = SummaryWriter('runs')
            tb_tracker = TensorBoardGradientTracker(writer)

            model = Model(...)
            model.fit_generator(..., callbacks=[tb_tracker])
    """

    def __init__(self, writer, keep_bias: bool = False) -> None:
        super().__init__(keep_bias)

        self.writer = writer

    def _on_epoch_end_log(self, epoch_number: int, logs: Dict) -> None:
        gradient_distributions_stats = ["mean", "mean_std_dev_up", "mean_std_dev_down"]
        other_gradient_stats = ["min", "max"]

        formatted_stats = self.tracker.get_stats(self.layer_names)
        for layer_name in self.layer_names:
            stats = formatted_stats[layer_name]

            for gradient_distributions_stat in gradient_distributions_stats:
                self.writer.add_scalars(
                    f'gradient_distributions/{layer_name}',
                    {gradient_distributions_stat: stats[gradient_distributions_stat]},
                    epoch_number,
                )
            for other_gradient_stat in other_gradient_stats:
                self.writer.add_scalars(
                    f'other_gradient_stats/{layer_name}',
                    {other_gradient_stat: stats[other_gradient_stat]},
                    epoch_number,
                )
