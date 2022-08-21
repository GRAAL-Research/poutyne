"""
This script can be simply ran with:
python basic_random_regression_with_experiment.py

Look in ./saves/my_regression_network for the checkpoints and logging.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score

# Import the Poutyne Model
from poutyne import ModelBundle, SKLearnMetrics

# Define a random toy dataset
num_features = 20
hidden_state_size = 100

num_train_samples = 800
train_x = np.random.randn(num_train_samples, num_features).astype('float32')
train_y = np.random.randn(num_train_samples, 1).astype('float32')

num_valid_samples = 200
valid_x = np.random.randn(num_valid_samples, num_features).astype('float32')
valid_y = np.random.randn(num_valid_samples, 1).astype('float32')

num_test_samples = 200
test_x = np.random.randn(num_test_samples, num_features).astype('float32')
test_y = np.random.randn(num_test_samples, 1).astype('float32')

cuda_device = 0
device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')

# Define the network
network = nn.Sequential(
    nn.Linear(num_features, hidden_state_size),
    nn.ReLU(),
    nn.Linear(hidden_state_size, 1),
)

# Everything is saved in ./saves/my_regression_network
model_bundle = ModelBundle.from_network(
    './saves/my_regression_network',
    network,
    device=device,
    optimizer='sgd',
    task='regression',
    batch_metrics=['l1'],
    epoch_metrics=[SKLearnMetrics(r2_score)],
)

model_bundle.train_data(train_x, train_y, validation_data=(valid_x, valid_y), epochs=5)

model_bundle.test_data(test_x, test_y)
