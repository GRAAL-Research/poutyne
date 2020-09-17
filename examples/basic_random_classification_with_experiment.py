"""
This script can be simply ran with:
python basic_random_classification_with_experiment.py

Look in ./expt/my_classification_network for the checkpoints and logging.
"""

# Import the Poutyne Model
from poutyne import Experiment, TensorDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Define a random toy dataset
num_features = 20
num_classes = 5
hidden_state_size = 100

num_train_samples = 800
train_x = np.random.randn(num_train_samples, num_features).astype('float32')
train_y = np.random.randint(num_classes, size=num_train_samples).astype('int64')

num_valid_samples = 200
valid_x = np.random.randn(num_valid_samples, num_features).astype('float32')
valid_y = np.random.randint(num_classes, size=num_valid_samples).astype('int64')

num_test_samples = 200
test_x = np.random.randn(num_test_samples, num_features).astype('float32')
test_y = np.random.randint(num_classes, size=num_test_samples).astype('int64')

cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

# Define the network
network = nn.Sequential(
    nn.Linear(num_features, hidden_state_size),
    nn.ReLU(),
    nn.Linear(hidden_state_size, num_classes)
)

# We need to use dataloaders (i.e. an iterable of batches) with Experiment
train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32)
valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=32)
test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

# Everything is saved in ./expt/my_classification_network
expt = Experiment('./expt/my_classification_network', network, device=device, optimizer='sgd', task='classif')

expt.train(train_loader, valid_loader, epochs=5)

expt.test(test_loader)
