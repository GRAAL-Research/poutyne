"""
Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.
"""

# pylint: disable=unused-argument

import numpy as np
import torch

from poutyne import Metric

some_metric_1_value = 1.0
some_metric_2_value = 2.0
repeat_batch_metric_value = 3.0

some_constant_metric_value = 3


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
        return (
            (
                np.random.rand(self.batch_size, 1).astype(np.float32),
                np.random.rand(self.batch_size, 1).astype(np.float32),
            )
            for _ in range(self.length)
        )


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


class SomeConstantMetric(Metric):
    def update(self, y_pred, y_true):
        pass

    def compute(self):
        return torch.FloatTensor([some_constant_metric_value])

    def reset(self):
        pass


class SomeMetric(Metric):
    def __init__(self):
        super().__init__()
        self.increment = 0.0

    def update(self, y_pred, y_true):
        self.increment += 1

    def compute(self):
        increment_value = self.increment
        self.increment = 0
        return increment_value

    def reset(self):
        pass


class SomeBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.increment = 0.0

    def forward(self, y_pred, y_true):
        self.increment += 1
        return self.increment

    def compute(self):
        increment_value = self.increment
        self.increment = 0
        return increment_value

    def reset(self):
        pass
