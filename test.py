import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

from poutyne import set_seeds
from poutyne.framework import Experiment, SummaryWriter
from poutyne.framework.callbacks.gradient_tracking import GradientTracker

set_seeds(42)

cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

# The dataset is split 80/20 for the train and validation datasets respectively.
train_split_percent = 0.8

# The MNIST dataset has 10 classes
num_classes = 10

# Training hyperparameters
batch_size = 32
learning_rate = 0.1
num_epochs = 5

full_train_dataset = MNIST('./mnist/', train=True, download=True, transform=transforms.ToTensor())
test_dataset = MNIST('./mnist/', train=False, download=True, transform=transforms.ToTensor())

num_data = len(full_train_dataset)
indices = list(range(num_data))
np.random.shuffle(indices)

split = math.floor(train_split_percent * num_data)

train_indices = indices[:split]
train_dataset = Subset(full_train_dataset, train_indices)

valid_indices = indices[split:]
valid_dataset = Subset(full_train_dataset, valid_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

loaders = train_loader, valid_loader, test_loader


def create_fully_connected_network():
    """
    This function returns the fully-connected network layed out above.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes)
    )


def create_convolutional_network():
    """
    This function returns the convolutional network layed out above.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )


conv_net = create_convolutional_network()
epochs = 10

callback = [GradientTracker(writer=SummaryWriter("./conv_net_experiment/tensorboard"))]
callback = []
exp = Experiment('./conv_net_experiment', conv_net, device=device, optimizer='sgd', task='classif')

exp.train(train_loader, valid_loader, epochs=epochs, callbacks=callback)

exp.test(test_loader)
