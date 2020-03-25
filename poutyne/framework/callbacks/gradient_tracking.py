import torch
from torch.tensor import Tensor

from poutyne.framework import Callback


class GradientTracker(Callback):
    def __init__(self, writer, keep_bias=False):
        super().__init__()
        self.writer = writer
        self.keep_bias = keep_bias

        self.layer_names = []
        self.number_layers = 0

        self.running_mean = None
        self.running_variance = None
        self.running_m2 = None  # used to calculate the variance
        self.running_min = None
        self.running_max = None

    def on_train_begin(self, logs):
        for layer_name, layer_params in self.model.model.named_parameters():
            if self._keep_layer(layer_params, layer_name):
                self.layer_names.append(layer_name)

        self.number_layers = len(self.layer_names)

    def on_epoch_begin(self, epoch, logs):
        self.running_mean = torch.zeros([self.number_layers])
        self.running_variance = torch.zeros([self.number_layers])
        self.running_m2 = torch.zeros([self.number_layers])
        self.running_min = torch.zeros([self.number_layers])
        self.running_max = torch.zeros([self.number_layers])

    def on_batch_end(self, batch, logs):
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

        batch_layer_means = Tensor(batch_layer_means)
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
