import os
from unittest import TestCase

import torch
from torch import nn

from poutyne import Experiment

test_checkpoints_path = "./tests_checkpoints"
ckpt_1_path = os.path.join(test_checkpoints_path, "checkpoint_epoch_1.ckpt")
ckpt_last_path = os.path.join(test_checkpoints_path, "checkpoint.ckpt")


def some_data_tensor_generator(batch_size):
    for _ in range(100):
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y


class ExperimentTest(TestCase):

    def setUp(self):
        self.pytorch_network = nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.loss_function = nn.MSELoss()

        self.test_experiment = Experiment(test_checkpoints_path,
                                          self.pytorch_network,
                                          optimizer=self.optimizer,
                                          loss_function=self.loss_function,
                                          monitor_metric="loss",
                                          monitor_mode="max")

        train_generator = some_data_tensor_generator(2)
        self.test_experiment.train(train_generator, epochs=1)

    def test_load_checkpoint_with_int(self):
        self.test_experiment.load_checkpoint(1)

        self.assertEqual(self.test_experiment.model.network.state_dict(), torch.load(ckpt_1_path, map_location="cpu"))

    def test_load_checkpoint_best(self):
        self.test_experiment.load_checkpoint("best")

        self.assertEqual(self.test_experiment.model.network.state_dict(), torch.load(ckpt_1_path, map_location="cpu"))

    def test_load_checkpoint_last(self):
        self.test_experiment.load_checkpoint("last")

        self.assertEqual(self.test_experiment.model.network.state_dict(), torch.load(ckpt_last_path,
                                                                                     map_location="cpu"))

    def test_load_checkpoint_path_state_dict(self):
        cpkt_path = os.path.join(test_checkpoints_path, "test_model_weights_state_dict.p")
        torch.save(torch.load(ckpt_1_path, map_location="cpu"), cpkt_path)  # change the ckpt path
        self.test_experiment.load_checkpoint(cpkt_path)

        self.assertEqual(self.test_experiment.model.network.state_dict(), torch.load(cpkt_path, map_location="cpu"))
