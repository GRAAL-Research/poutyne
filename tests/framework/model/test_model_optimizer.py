import unittest
import torch
import torch.nn as nn
import torch.optim as optim

from poutyne import Model


class ModelOptimizerInstanciationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()

    def test_with_optimizer_object(self):
        optimizer = optim.SGD(self.pytorch_network.parameters(), lr=0.1)
        model = Model(self.pytorch_network, optimizer, self.loss_function)
        self.assertOptimizerEquality(model.optimizer, optimizer)

    def test_with_string_optimizer(self):
        model = Model(self.pytorch_network, 'sgd', self.loss_function)
        expected = optim.SGD(self.pytorch_network.parameters(), lr=1e-2)
        self.assertOptimizerEquality(model.optimizer, expected)

    def test_with_dict_optimizer(self):
        model = Model(self.pytorch_network, dict(optim='sgd', lr=1e-3), self.loss_function)
        expected = optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.assertOptimizerEquality(model.optimizer, expected)

    def test_with_other_string_than_sgd(self):
        model = Model(self.pytorch_network, 'adam', self.loss_function)
        expected = optim.Adam(self.pytorch_network.parameters())
        self.assertOptimizerEquality(model.optimizer, expected)

    def test_with_other_dict_than_sgd(self):
        model = Model(self.pytorch_network, dict(optim='adam', lr=0.1), self.loss_function)
        expected = optim.Adam(self.pytorch_network.parameters(), lr=0.1)
        self.assertOptimizerEquality(model.optimizer, expected)

    def test_with_string_optimizer_with_parameters_that_requires_no_grad(self):
        self.pytorch_network.bias.requires_grad = False
        model = Model(self.pytorch_network, 'sgd', self.loss_function)
        expected = optim.SGD([self.pytorch_network.weight], lr=1e-2)
        self.assertOptimizerEquality(model.optimizer, expected)

    def assertOptimizerEquality(self, first, second):
        self.assertEqual(first.state_dict()['param_groups'], second.state_dict()['param_groups'])
