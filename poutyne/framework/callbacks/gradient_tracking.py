import numpy as np
from torch.utils.tensorboard import SummaryWriter

from poutyne.framework import Callback


class GradientTracker(Callback):
    """
    args:
        logging_path (str): The path to log the Tensorboard entries. If you want to log the Tensorboard entries in the
        same Tensorboard entries as the :class:`~poutyne.framework.callbacks.logger.TensorBoardLogger` you need to use
        the same logging directory as the one used for the experiment appended with "tensorboard"
        (e.g. `"./logging/tensorboard"`).
        keep_bias (bool): Either or not to log the bias of the network.

    """

    def __init__(self, logging_path: str, keep_bias: bool = False) -> None:
        super().__init__()

        self.writer = SummaryWriter(logging_path)
        self.keep_bias = keep_bias

        self.layer_names = []
        self.number_layers = 0

        self.running_mean = None
        self.running_variance = None
        self.running_m2 = None  # used to calculate the variance
        self.running_min = None
        self.running_max = None

    def on_epoch_begin(self, epoch, logs):
        self.running_mean = np.zeros([self.number_layers])
        self.running_variance = np.zeros([self.number_layers])
        self.running_m2 = np.zeros([self.number_layers])
        self.running_min = np.zeros([self.number_layers])
        self.running_max = np.zeros([self.number_layers])

    def on_epoch_end(self, epoch, logs):
        for index, layer_name in enumerate(self.layer_names):
            graph_name = "gradient_distributions/" + layer_name
            self.writer.add_scalars(graph_name, {"mean": self.running_mean[index]}, epoch)
            self.writer.add_scalars(graph_name,
                                    {"std_dev_up": self.running_mean[index] + np.sqrt(self.running_variance[index])},
                                    epoch)
            self.writer.add_scalars(
                graph_name, {"std_dev_down": self.running_mean[index] - np.sqrt(self.running_variance[index])},
                epoch)
            graph_name = "other_gradient_stats/" + layer_name
            self.writer.add_scalars(graph_name, {"min": self.running_min[index]}, epoch)
            self.writer.add_scalars(graph_name, {"max": self.running_max[index]}, epoch)

    def on_train_batch_end(self, batch, logs):
        self._on_batch_end_write(batch)

    def _on_batch_end_write(self, batch):
        batch_layer_means = []
        batch_layer_min = []
        batch_layer_max = []
        for layer_name, layer_params in self.model.model.named_parameters():
            if self._keep_layer(layer_params, layer_name):
                layer_gradient = layer_params.grad
                abs_value_layer_gradient = layer_gradient.abs()

                batch_layer_means.append(abs_value_layer_gradient.mean())
                batch_layer_min.append(layer_gradient.min())
                batch_layer_max.append(layer_gradient.max())

        batch_layer_means = np.array(batch_layer_means)
        self.previous_mean = self.running_mean

        self.running_mean = self.previous_mean + (batch_layer_means - self.previous_mean) / batch

        self.running_m2 = self.running_m2 + (batch_layer_means - self.previous_mean) * (batch_layer_means -
                                                                                        self.running_mean)

        self.running_variance = self.running_m2 / (batch - 1) if batch > 1 else self.running_variance

        batch_layer_min = np.array(batch_layer_min)
        batch_layer_max = np.array(batch_layer_max)

        self.running_min = np.min(batch_layer_min, self.running_min)
        self.running_max = np.max(batch_layer_max, self.running_max)

    def on_train_begin(self, logs):
        for layer_name, layer_params in self.model.model.named_parameters():
            if self._keep_layer(layer_params, layer_name):
                self.layer_names.append(layer_name)

        self.number_layers = len(self.layer_names)

    def on_train_end(self, logs):
        self.writer.close()

    def _keep_layer(self, layer_params, layer_name):
        layer_require_grad = layer_params.requires_grad
        if self.keep_bias:
            return layer_require_grad
        else:
            return layer_require_grad and ("bias" not in layer_name)
