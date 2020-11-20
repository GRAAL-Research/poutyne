# Import the Poutyne Model
from poutyne import Model
import torch
import torch.nn as nn
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

# Train
model = Model(network, 'sgd', 'cross_entropy',
              batch_metrics=['accuracy'],
              epoch_metrics=['f1'],
              device=device)
model.fit(
    train_x, train_y,
    validation_data=(valid_x, valid_y),
    epochs=5,
    batch_size=32
)
