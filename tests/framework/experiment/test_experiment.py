import os
from unittest import TestCase

from tempfile import TemporaryDirectory

import torch
from torch import nn

from poutyne import Experiment


def some_data_tensor_generator(batch_size):
    for _ in range(100):
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y


class ExperimentTest(TestCase):

    def setUp(self):
        self.temp_dir_obj = TemporaryDirectory()
        self.test_checkpoints_path = self.temp_dir_obj.name
        self.ckpt_1_path = os.path.join(self.test_checkpoints_path, "checkpoint_epoch_1.ckpt")
        self.ckpt_last_path = os.path.join(self.test_checkpoints_path, "checkpoint.ckpt")

        expt = Experiment(self.test_checkpoints_path,
                          nn.Linear(1, 1),
                          optimizer='sgd',
                          loss_function='mse',
                          monitor_metric="loss",
                          monitor_mode="min")
        train_generator = some_data_tensor_generator(2)
        expt.train(train_generator, epochs=1)

        self.test_experiment = Experiment(self.test_checkpoints_path,
                                          nn.Linear(1, 1),
                                          optimizer='sgd',
                                          loss_function='mse',
                                          monitor_metric="loss",
                                          monitor_mode="min")

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_load_checkpoint_with_int(self):
        self.test_experiment.load_checkpoint(1)

        self.assertEqual(self.test_experiment.model.network.state_dict(),
                         torch.load(self.ckpt_1_path, map_location="cpu"))

    def test_load_checkpoint_best(self):
        self.test_experiment.load_checkpoint("best")

        self.assertEqual(self.test_experiment.model.network.state_dict(),
                         torch.load(self.ckpt_1_path, map_location="cpu"))

    def test_load_checkpoint_last(self):
        self.test_experiment.load_checkpoint("last")

        self.assertEqual(self.test_experiment.model.network.state_dict(),
                         torch.load(self.ckpt_last_path, map_location="cpu"))

    def test_load_checkpoint_path_state_dict(self):
        cpkt_path = os.path.join(self.test_checkpoints_path, "test_model_weights_state_dict.p")
        torch.save(torch.load(self.ckpt_1_path, map_location="cpu"), cpkt_path)  # change the ckpt path
        self.test_experiment.load_checkpoint(cpkt_path)

        self.assertEqual(self.test_experiment.model.network.state_dict(), torch.load(cpkt_path, map_location="cpu"))
