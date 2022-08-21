import torch
import torch.nn as nn
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from poutyne import Model, Metric

# Instanciate the MNIST dataset
train_valid_dataset = MNIST('./datasets', train=True, download=True, transform=ToTensor())
test_dataset = MNIST('./datasets', train=False, download=True, transform=ToTensor())
train_dataset, valid_dataset = random_split(
    train_valid_dataset, [50_000, 10_000], generator=torch.Generator().manual_seed(42)
)

# Select CUDA device if available
cuda_device = 0
device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')

# Define the network
network = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)
epochs = 5


# Creating a metric using the PyTorch loss interface.
def my_accuracy_func(y_pred, y_true):
    y_pred = y_pred.argmax(1)
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100


# Creating a metric using the Metric interface.
class MyAccuracyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.reset()

    def _compute_stats(self, y_pred, y_true):
        """
        Compute the number of true positives and the total number of elements
        of a given batch.
        """
        y_pred = y_pred.argmax(1)
        num_true_positives = (y_pred == y_true).long().sum().item()
        total_exemples = y_true.numel()
        return num_true_positives, total_exemples

    def _compute_accuracy_from_stats(self, num_true_positives, total_exemples):
        """
        Compute the accuracy given by the statistics computed in _compute_stats().
        """
        return num_true_positives / total_exemples * 100

    def _update(self, y_pred, y_true):
        """
        Increment the running number of true positives and number of elements.
        """
        num_true_positives, total_exemples = self._compute_stats(y_pred, y_true)
        self.num_true_positives += num_true_positives
        self.total_exemples += total_exemples
        return num_true_positives, total_exemples

    def forward(self, y_pred, y_true):
        """
        When using the metric as a batch metric, this is called. If the metric is only
        used as an epoch metric, there is no need to implement this.
        """
        num_true_positives, total_exemples = self._update(y_pred, y_true)
        return self._compute_accuracy_from_stats(num_true_positives, total_exemples)

    def update(self, y_pred, y_true):
        """
        When using the metric as an epoch metric, this is called. If the metric is only
        used as a batch metric, there is no need to implement this.
        """
        self._update(y_pred, y_true)

    def compute(self):
        """
        This is called at the end to get the value for the whole epoch.
        """
        return self._compute_accuracy_from_stats(self.num_true_positives, self.total_exemples)

    def reset(self) -> None:
        """
        Reset the statistics for another epoch.
        """
        self.num_true_positives = 0
        self.total_exemples = 0


# Define the Model and train with our custom metrics
model = Model(
    network,
    'sgd',
    'cross_entropy',
    batch_metrics=[my_accuracy_func, MyAccuracyMetric()],
    device=device,
)
model.fit_dataset(train_dataset, valid_dataset, epochs=epochs)
