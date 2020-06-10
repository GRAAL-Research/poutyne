import torch.nn as nn
import torch.optim as optim
from poutyne.framework import Model

net = nn.Sequential(
    nn.Linear(100, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

model = Model(net, optimizer, criterion, batch_metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5,  batch_size=32)
loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
