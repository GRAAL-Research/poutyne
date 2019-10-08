import torch
from torch import tensor

from poutyne.framework import Callback


class GradientTracker(Callback):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        self.layers_name = []
        self.number_layers = 0

        self.running_mean = None
        self.running_variance = None
        self.running_m2 = None  # use to calculate the variance
        self.running_min = None
        self.running_max = None

    def on_train_begin(self, logs):
        for layer_name, layer_gradient in self.model.model.named_parameters():
            if layer_gradient.requires_grad and ("bias" not in layer_name):
                self.layers_name.append(layer_name)

        self.number_layers = len(self.layers_name)

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
        for layer_name, layer_gradient in self.model.model.named_parameters():
            if layer_gradient.requires_grad and ("bias" not in layer_name):
                batch_layer_means.append(layer_gradient.grad.abs().mean())
                batch_layer_min.append(layer_gradient.grad.abs().min())
                batch_layer_max.append(layer_gradient.grad.abs().max())

        batch_layer_means = tensor(batch_layer_means)
        self.previous_mean = self.running_mean

        self.running_mean = self.previous_mean + (batch_layer_means - self.previous_mean) / batch

        self.running_m2 = self.running_m2 + (batch_layer_means - self.previous_mean) * (
                batch_layer_means - self.running_mean)

        self.running_variance = self.running_m2 / (batch - 1) if batch > 1 else self.running_variance

        batch_layer_min = tensor(batch_layer_min)
        batch_layer_max = tensor(batch_layer_max)

        self.running_min = torch.cat((batch_layer_min.unsqueeze(1), self.running_min.unsqueeze(1)), dim=-1).max(
            dim=1).values
        self.running_max = torch.cat((batch_layer_max.unsqueeze(1), self.running_max.unsqueeze(1)), dim=-1).max(
            dim=1).values

    def on_epoch_end(self, epoch, logs):
        self._on_epoch_end_write(epoch)

    def _on_epoch_end_write(self, epoch):
        for index, layer_name in enumerate(self.layers_name):
            graph_name = "_".join([layer_name, "gradient_distribution"])
            self.writer.add_scalars(graph_name, {"mean": self.running_mean[index]}, epoch)
            self.writer.add_scalars(graph_name,
                                    {"std_dev_up": self.running_mean[index] + torch.sqrt(self.running_variance[index])},
                                    epoch)
            self.writer.add_scalars(graph_name,
                                    {"std_dev_down": self.running_mean[index] - torch.sqrt(
                                        self.running_variance[index])},
                                    epoch)
            graph_name = "_".join([layer_name, "other_gradient_stats"])
            self.writer.add_scalars(graph_name, {"min": self.running_min[index]}, epoch)
            self.writer.add_scalars(graph_name, {"max": self.running_max[index]}, epoch)

    def on_train_end(self, logs=None):
        pass
