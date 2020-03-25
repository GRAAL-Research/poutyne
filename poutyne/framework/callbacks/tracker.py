from typing import Dict, List

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .callbacks import Callback


class GradientStatsTracker:
    def __init__(self, keep_bias):
        self.keep_bias = keep_bias

        self.layer_names = []
        self.number_layers = 0

        self.running_mean = None
        self.running_variance = None
        self.running_m2 = None  # used to calculate the variance
        self.running_min = None
        self.running_max = None

    def on_epoch_begin_init(self):
        self.running_mean = np.zeros([self.number_layers], dtype="float32")
        self.running_variance = np.zeros([self.number_layers], dtype="float32")
        self.running_m2 = np.zeros([self.number_layers], dtype="float32")
        self.running_min = np.zeros([self.number_layers], dtype="float32")
        self.running_max = np.zeros([self.number_layers], dtype="float32")

    def number_layer_compute(self, layer_name, layer_params):
        if self._keep_layer(layer_params, layer_name):
            self.layer_names.append(layer_name)

        self.number_layers = len(self.layer_names)

    def upgrade_batch_grad(self, named_parameters, batch):
        batch_layer_means = []
        batch_layer_min = []
        batch_layer_max = []
        for layer_name, layer_params in named_parameters:
            if self._keep_layer(layer_params, layer_name):
                layer_gradient = layer_params.grad
                abs_value_layer_gradient = layer_gradient.abs()

                batch_layer_means.append(abs_value_layer_gradient.mean().cpu().detach().numpy())
                batch_layer_min.append(layer_gradient.min().cpu().detach().numpy())
                batch_layer_max.append(layer_gradient.max().cpu().detach().numpy())

        batch_layer_means = np.array(batch_layer_means)
        previous_mean = self.running_mean

        self.running_mean = previous_mean + (batch_layer_means - previous_mean) / batch

        self.running_m2 = self.running_m2 + (batch_layer_means - previous_mean) * (batch_layer_means -
                                                                                   self.running_mean)

        self.running_variance = self.running_m2 / (batch - 1) if batch > 1 else self.running_variance

        batch_layer_min = np.array(batch_layer_min)
        batch_layer_max = np.array(batch_layer_max)

        self.running_min = np.minimum(batch_layer_min, self.running_min)
        self.running_max = np.maximum(batch_layer_max, self.running_max)

    def get_stats(self) -> List:
        formatted_stats = []
        for index, layer_name in enumerate(self.layer_names):
            stats = [("mean", self.running_mean[index]),
                     ("std_dev_up", self.running_mean[index] + np.sqrt(self.running_variance[index])),
                     ("std_dev_down", self.running_mean[index] - np.sqrt(self.running_variance[index]))]

            formatted_stats.append(("gradient_distributions/{}".format(layer_name), stats))

            other_stats = [("min", self.running_min[index]), ("max", self.running_max[index])]

            formatted_stats.append(("other_gradient_stats/{}".format(layer_name), other_stats))
        return formatted_stats

    def _keep_layer(self, layer_params, layer_name):
        layer_require_grad = layer_params.requires_grad
        if self.keep_bias:
            return layer_require_grad
        return layer_require_grad and ("bias" not in layer_name)


class TensorBoardGradientTracker(Callback):
    """
    args:
        writer (~torch.utils.tensorboard.writer.SummaryWriter): The TensorBoard writer.
        keep_bias (bool): Either or not to log the bias of the network.
    """

    def __init__(self, writer: SummaryWriter, keep_bias: bool = False) -> None:
        super().__init__()

        self.writer = writer

        self.tracker = GradientStatsTracker(keep_bias=keep_bias)

    def on_epoch_begin(self, epoch, logs):
        # pylint: disable=unused-argument
        self.tracker.on_epoch_begin_init()

    def on_epoch_end(self, epoch, logs):
        formatted_stats = self.tracker.get_stats()
        for layer_name, stats in formatted_stats:
            for stat_name, value in stats:
                self.writer.add_scalars(layer_name, {stat_name: value}, epoch)

    def on_train_batch_end(self, batch, logs):
        self.tracker.upgrade_batch_grad(self.model.network.named_parameters(), batch)

    def on_train_begin(self, logs):
        for layer_name, layer_params in self.model.network.named_parameters():
            self.tracker.number_layer_compute(layer_name, layer_params)

    def on_train_end(self, logs):
        self.writer.close()
