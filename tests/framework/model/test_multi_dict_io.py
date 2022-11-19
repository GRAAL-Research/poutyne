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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from poutyne import Model

from .base import ModelFittingTestCase


class DictIOModel(nn.Module):
    """Model to test multiple dict input/output"""

    def __init__(self, input_keys, output_keys):
        super().__init__()
        assert len(input_keys) == len(output_keys)
        inputs = {k: nn.Linear(1, 1) for k in input_keys}
        self.inputs = nn.ModuleDict(inputs)

        self.input_keys = input_keys
        self.output_keys = output_keys

    def forward(self, x):
        return {out_k: self.inputs[in_k](x[in_k]) for in_k, out_k in zip(self.input_keys, self.output_keys)}


def dict_mse_loss(y_pred, y_true):
    return sum(F.mse_loss(y_pred[k], y_true[k]) for k in y_true.keys())


def get_batch(batch_size):
    x1 = torch.rand(batch_size, 1)
    x2 = torch.rand(batch_size, 1)
    y1 = torch.rand(batch_size, 1)
    y2 = torch.rand(batch_size, 1)
    return dict(x1=x1, x2=x2), dict(y1=y1, y2=y2)


def some_data_tensor_generator_dict_io(batch_size):
    while True:
        yield get_batch(batch_size)


class ModelMultiDictIOTest(ModelFittingTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.pytorch_network = DictIOModel(['x1', 'x2'], ['y1', 'y2'])
        self.loss_function = dict_mse_loss
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)

        self.model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics,
        )

    def test_fitting_tensor_generator_multi_dict_io(self):
        train_generator = some_data_tensor_generator_dict_io(ModelMultiDictIOTest.batch_size)
        valid_generator = some_data_tensor_generator_dict_io(ModelMultiDictIOTest.batch_size)
        logs = self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelMultiDictIOTest.epochs,
            steps_per_epoch=ModelMultiDictIOTest.steps_per_epoch,
            validation_steps=ModelMultiDictIOTest.steps_per_epoch,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelMultiDictIOTest.epochs,
            'steps': ModelMultiDictIOTest.steps_per_epoch,
            'valid_steps': ModelMultiDictIOTest.steps_per_epoch,
        }
        self._test_callbacks_train(params, logs)

    def test_tensor_train_on_batch_multi_dict_io(self):
        x, y = get_batch(ModelMultiDictIOTest.batch_size)
        loss = self.model.train_on_batch(x, y)
        self.assertEqual(type(loss), float)

    def test_train_on_batch_with_pred_multi_dict_io(self):
        x, y = get_batch(ModelMultiDictIOTest.batch_size)
        loss, pred_y = self.model.train_on_batch(x, y, return_pred=True)
        self.assertEqual(type(loss), float)
        for value in pred_y.values():
            self.assertEqual(value.shape, (ModelMultiDictIOTest.batch_size, 1))

    def test_ndarray_train_on_batch_multi_dict_io(self):
        x1 = np.random.rand(ModelMultiDictIOTest.batch_size, 1).astype(np.float32)
        x2 = np.random.rand(ModelMultiDictIOTest.batch_size, 1).astype(np.float32)
        y1 = np.random.rand(ModelMultiDictIOTest.batch_size, 1).astype(np.float32)
        y2 = np.random.rand(ModelMultiDictIOTest.batch_size, 1).astype(np.float32)
        x, y = dict(x1=x1, x2=x2), dict(y1=y1, y2=y2)
        loss = self.model.train_on_batch(x, y)
        self.assertEqual(type(loss), float)

    def test_evaluate_generator_multi_dict_io(self):
        num_steps = 10
        generator = some_data_tensor_generator_dict_io(ModelMultiDictIOTest.batch_size)
        loss, pred_y = self.model.evaluate_generator(generator, steps=num_steps, return_pred=True)
        self.assertEqual(type(loss), float)
        self._test_size_and_type_for_generator(pred_y, (num_steps * ModelMultiDictIOTest.batch_size, 1))

    def test_tensor_evaluate_on_batch_multi_dict_io(self):
        x, y = get_batch(ModelMultiDictIOTest.batch_size)
        loss = self.model.evaluate_on_batch(x, y)
        self.assertEqual(type(loss), float)

    def test_predict_generator_multi_dict_io(self):
        num_steps = 10
        generator = some_data_tensor_generator_dict_io(ModelMultiDictIOTest.batch_size)
        generator = (x for x, _ in generator)
        pred_y = self.model.predict_generator(generator, steps=num_steps)
        self._test_size_and_type_for_generator(pred_y, (num_steps * ModelMultiDictIOTest.batch_size, 1))

    def test_tensor_predict_on_batch_multi_dict_io(self):
        x1 = torch.rand(ModelMultiDictIOTest.batch_size, 1)
        x2 = torch.rand(ModelMultiDictIOTest.batch_size, 1)
        pred_y = self.model.predict_on_batch(dict(x1=x1, x2=x2))
        self._test_size_and_type_for_generator(pred_y, (ModelMultiDictIOTest.batch_size, 1))
