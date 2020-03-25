# Because of the way the callbacks are generated, we have to disable linting here.
# pylint: disable=no-name-in-module
import unittest
from unittest import TestCase

import torch
import torch.nn as nn

from poutyne.framework import Model
from poutyne.framework.callbacks import LambdaLR, StepLR, MultiStepLR, ExponentialLR, \
    CosineAnnealingLR, ReduceLROnPlateau


def some_data_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y


class LRSchedulersTest(TestCase):
    batch_size = 20
    epochs = 10
    steps_per_epoch = 5

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.train_gen = some_data_generator(20)
        self.valid_gen = some_data_generator(20)

    def test_lambda_lr_integration(self):
        my_lambda = lambda epoch: 0.95**epoch
        lambda_lr = LambdaLR(lr_lambda=[my_lambda])
        self._fit_with_callback_integration(lambda_lr)

    def test_step_lr_integration(self):
        step_lr = StepLR(step_size=3)
        self._fit_with_callback_integration(step_lr)

    def test_multistep_lr_integration(self):
        multistep_lr = MultiStepLR(milestones=[2, 5, 7])
        self._fit_with_callback_integration(multistep_lr)

    def test_exponential_lr_integration(self):
        exponential_lr = ExponentialLR(gamma=0.01)
        self._fit_with_callback_integration(exponential_lr)

    def test_cosine_annealing_lr_integration(self):
        cosine_annealing_lr = CosineAnnealingLR(T_max=8)
        self._fit_with_callback_integration(cosine_annealing_lr)

    def test_reduce_lr_on_plateau_integration(self):
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=3)
        self._fit_with_callback_integration(reduce_lr)

    def _fit_with_callback_integration(self, callback):
        self.model.fit_generator(self.train_gen,
                                 self.valid_gen,
                                 epochs=LRSchedulersTest.epochs,
                                 steps_per_epoch=LRSchedulersTest.steps_per_epoch,
                                 callbacks=[callback])

    def test_exception_is_thrown_on_optimizer_argument(self):
        with self.assertRaises(ValueError):
            StepLR(self.optimizer, step_size=3)


if __name__ == '__main__':
    unittest.main()
