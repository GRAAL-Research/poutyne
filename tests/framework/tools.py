# pylint: disable=unused-argument

import numpy as np
import torch

from poutyne import EpochMetric

some_metric_1_value = 1.
some_metric_2_value = 2.
repeat_batch_metric_value = 3.

some_constant_epoch_metric_value = 3


def some_data_tensor_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y


def some_data_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y


class SomeDataGeneratorUsingStopIteration:

    def __init__(self, batch_size, length):
        self.batch_size = batch_size
        self.length = length

    def __iter__(self):
        return ((np.random.rand(self.batch_size, 1).astype(np.float32), np.random.rand(self.batch_size,
                                                                                       1).astype(np.float32))
                for _ in range(self.length))


class SomeDataGeneratorWithLen:

    def __init__(self, batch_size, length, num_missing_samples):
        self.batch_size = batch_size
        self.length = length
        self.num_generator_called = 0
        self.x = torch.rand(length * batch_size - num_missing_samples, 1)
        self.y = torch.rand(length * batch_size - num_missing_samples, 1)

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        start_idx = self.num_generator_called * self.batch_size
        end_idx = (self.num_generator_called + 1) * self.batch_size
        x = self.x[start_idx:end_idx]
        y = self.y[start_idx:end_idx]
        self.num_generator_called += 1
        if self.num_generator_called == self.length:
            self.num_generator_called = 0
        return x, y


def some_batch_metric_1(y_pred, y_true):
    return torch.FloatTensor([some_metric_1_value])


def some_batch_metric_2(y_pred, y_true):
    return torch.FloatTensor([some_metric_2_value])


def repeat_batch_metric(y_pred, y_true):
    return torch.FloatTensor([repeat_batch_metric_value])


class SomeConstantEpochMetric(EpochMetric):

    def forward(self, y_pred, y_true):
        pass

    def get_metric(self):
        return torch.FloatTensor([some_constant_epoch_metric_value])

    def reset(self):
        pass


class SomeEpochMetric(EpochMetric):

    def __init__(self):
        super().__init__()
        self.increment = 0.0

    def forward(self, y_pred, y_true):
        self.increment += 1

    def get_metric(self):
        increment_value = self.increment
        self.increment = 0
        return increment_value

    def reset(self):
        pass
