# pylint: disable=not-callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

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

# Train in pure PyTorch
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=1e-2)
network = network.to(device)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

for epoch in range(epochs):
    train_loss = 0.0
    for train_inputs, train_labels in train_loader:
        train_inputs = train_inputs.to(device)
        train_labels = train_labels.to(device)

        optimizer.zero_grad()

        outputs = network(train_inputs)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()
    train_loss /= len(train_loader)

    val_loss = 0.0
    for val_inputs, val_labels in valid_loader:
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)

        with torch.no_grad():
            outputs = network(val_inputs)
            val_loss += criterion(outputs, val_labels)
    val_loss /= len(valid_loader)

    print(f'Epoch {epoch}: Train loss: {train_loss}, Valid loss: {val_loss}')
