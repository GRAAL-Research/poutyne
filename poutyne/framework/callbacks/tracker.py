from typing import Dict

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .callbacks import Callback


class Tracker(Callback):
    def __init__(self, keep_bias: bool = False) -> None:
        super().__init__()

        self.keep_bias = keep_bias
        self.writer = None

        self.layer_names = []
        self.number_layers = 0

        self.tracker = GradientStatsTracker()

    def on_epoch_begin(self, epoch, logs):
        # pylint: disable=unused-argument
        self.tracker.init(self.number_layers)

    def on_train_batch_end(self, batch, logs):
        for layer_name, layer_params in self.model.network.named_parameters():
            if self._keep_layer(layer_params, layer_name):
                layer_gradient = layer_params.grad
                self.tracker.upgrade_layer_batch_grad(layer_gradient)
        self.tracker.upgrade_batch_grad(batch)

    def on_train_begin(self, logs):
        for layer_name, layer_params in self.model.network.named_parameters():
            self._get_layers_to_track(layer_name, layer_params)

    def on_train_end(self, logs):
        self.writer.close()

    def _get_layers_to_track(self, layer_name, layer_params):
        if self._keep_layer(layer_params, layer_name):
            self.layer_names.append(layer_name)

        self.number_layers = len(self.layer_names)

    def _keep_layer(self, layer_params, layer_name):
        layer_require_grad = layer_params.requires_grad
        if self.keep_bias:
            return layer_require_grad
        return layer_require_grad and ("bias" not in layer_name)


class GradientStatsTracker:
    def __init__(self):
        self.running_mean = None
        self.running_variance = None
        self.running_m2 = None  # used to calculate the variance
        self.running_min = None
        self.running_max = None

        self.number_layers = None

        self.batch_layer_means = []
        self.batch_layer_min = []
        self.batch_layer_max = []

    def init(self, number_layers):
        self.number_layers = number_layers
        self.reset()

    def upgrade_layer_batch_grad(self, layer_gradient):
        abs_value_layer_gradient = layer_gradient.abs()

        self.batch_layer_means.append(abs_value_layer_gradient.mean().cpu().detach().numpy())
        self.batch_layer_min.append(layer_gradient.min().cpu().detach().numpy())
        self.batch_layer_max.append(layer_gradient.max().cpu().detach().numpy())

    def upgrade_batch_grad(self, batch):
        batch_layer_means = np.array(self.batch_layer_means)
        previous_mean = self.running_mean

        self.running_mean = previous_mean + (batch_layer_means - previous_mean) / batch

        self.running_m2 = self.running_m2 + (batch_layer_means - previous_mean) * (batch_layer_means -
                                                                                   self.running_mean)

        self.running_variance = self.running_m2 / (batch - 1) if batch > 1 else self.running_variance

        batch_layer_min = np.array(self.batch_layer_min)
        batch_layer_max = np.array(self.batch_layer_max)

        self.running_min = np.minimum(batch_layer_min, self.running_min)
        self.running_max = np.maximum(batch_layer_max, self.running_max)

        self._reset_batch_gradient_tracker()

    def get_stats(self, layer_names) -> Dict:
        formatted_stats = {}
        for index, layer_name in enumerate(layer_names):
            stats = {
                "mean": self.running_mean[index],
                "std_dev_up": self.running_mean[index] + np.sqrt(self.running_variance[index]),
                "std_dev_down": self.running_mean[index] - np.sqrt(self.running_variance[index]),
                "min": self.running_min[index],
                "max": self.running_max[index]
            }

            formatted_stats.update({layer_name: stats})

        self.reset()
        return formatted_stats

    def reset(self):
        self.running_mean = np.zeros([self.number_layers], dtype="float32")
        self.running_variance = np.zeros([self.number_layers], dtype="float32")
        self.running_m2 = np.zeros([self.number_layers], dtype="float32")
        self.running_min = np.zeros([self.number_layers], dtype="float32")
        self.running_max = np.zeros([self.number_layers], dtype="float32")

    def _reset_batch_gradient_tracker(self):
        self.batch_layer_means = []
        self.batch_layer_min = []
        self.batch_layer_max = []


class TensorBoardGradientTracker(Tracker):
    """
    args:
        writer (~torch.utils.tensorboard.writer.SummaryWriter): The TensorBoard writer.
        keep_bias (bool): Either or not to log the bias of the network.
    """

    def __init__(self, writer: SummaryWriter, keep_bias: bool = False) -> None:
        super().__init__(keep_bias)

        self.writer = writer

    def on_epoch_end(self, epoch, logs):
        gradient_distributions_stats = ["mean", "std_dev_up", "std_dev_down"]
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
