import torch
import torch.nn as nn
import torch.optim as optim
from poutyne import Model

net = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

num_features = 100
num_classes = 10

# Our training dataset with 8000 samples.
num_train_samples = 80000
x_train = torch.rand(num_train_samples, num_features)
y_train = torch.randint(num_classes, (num_train_samples,), dtype=torch.long)

# Our test dataset with 200 samples.
num_test_samples = 200
x_test = torch.rand(num_test_samples, num_features)
y_test = torch.randint(num_classes, (num_test_samples,), dtype=torch.long)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model = Model(net, optimizer, criterion, batch_metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5,  batch_size=32)
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
