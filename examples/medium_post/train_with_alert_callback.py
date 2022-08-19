import smtplib
from typing import Dict

import torch
import torch.nn as nn
from notif.notificator import EmailNotificator  # pip install notif
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from poutyne import Experiment, Callback


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


class TrainingAlertCallback(Callback):
    def __init__(
        self, sender_email: str, sender_login_credential: str, smtp_server: smtplib.SMTP, alert_frequency: int
    ):
        super().__init__()
        destination_email = sender_email  # sending e-mail to yourself in that case
        self.notif = EmailNotificator(sender_email, sender_login_credential, destination_email, smtp_server)

        self.alert_frequency = alert_frequency

    def on_train_begin(self, logs: Dict):
        self.notif.send_notification("Start of the training.")

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        if epoch_number % self.alert_frequency == 0:
            self.notif.send_notification(f"Epoch {epoch_number} is done.")

    def on_train_end(self, logs: Dict):
        self.notif.send_notification("End of the training.")


your_email = "your_email"
your_email_login_credential = "your_password"
email_provider = smtplib.SMTP('smtp.email_provider.com', 587)  # email_provider such as gmail or live # 587 for TLS port

# Alert at start/end of the training and at every epoch.
training_alert_callback = TrainingAlertCallback(
    your_email,
    your_email_login_credential,
    email_provider,
    alert_frequency=1,
)

# Define the Experiment and train
experiment = Experiment(
    './simple_model',  # Where to log
    network,
    optimizer='sgd',
    loss_function='cross_entropy',
    device=device,
)
experiment.train_dataset(train_dataset, valid_dataset, epochs=epochs, callbacks=[training_alert_callback])
