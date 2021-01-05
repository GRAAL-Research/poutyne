import torch
import torch.nn as nn
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from poutyne import Model, EpochMetric

# Instanciate the MNIST dataset
train_valid_dataset = MNIST('./datasets', train=True, download=True, transform=ToTensor())
test_dataset = MNIST('./datasets', train=False, download=True, transform=ToTensor())
train_dataset, valid_dataset = random_split(train_valid_dataset, [50_000, 10_000],
                                            generator=torch.Generator().manual_seed(42))

# Select CUDA device if available
cuda_device = 0
device = torch.device('cuda:%d' % cuda_device if torch.cuda.is_available() else 'cpu')

# Define the network
network = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)
epochs = 5

# Creating a batch metric for the accuracy
def my_accuracy(y_pred, y_true):
    y_pred = y_pred.argmax(1)
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100

# Creating an epoch metric for the accuracy
class MyEpochMetricAccuracy(EpochMetric):
    def __init__(self):
        super().__init__()
        self.reset()

    def forward(self, y_pred, y_true) -> None:
        # Increment the number of true positives and the total number of elemnts
        y_pred = y_pred.argmax(1)
        self.num_true_positives += (y_pred == y_true).long().sum().item()
        self.total_exemples += y_true.numel()

    def get_metric(self):
        # Compute the epoch metric with the numbers computed in forward
        return self.num_true_positives / self.total_exemples * 100

    def reset(self) -> None:
        # Reset the statistics for another epoch
        self.num_true_positives = 0
        self.total_exemples = 0

# Define the Model and train with our custom metrics
model = Model(network, 'sgd', 'cross_entropy',
              batch_metrics=[my_accuracy],
              epoch_metrics=[MyEpochMetricAccuracy()],
              device=device)
model.fit_dataset(train_dataset, valid_dataset, epochs=epochs)
