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
from torch.utils.data import DataLoader

from poutyne import Model, TensorDataset

from .base import ModelFittingTestCase, MultiIOModel


def some_data_tensor_generator_multi_input(batch_size):
    while True:
        x1 = torch.rand(batch_size, 1)
        x2 = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield (x1, x2), y


class ModelMultiInputTest(ModelFittingTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.pytorch_network = MultiIOModel(num_input=1, num_output=1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)

        self.model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics,
        )

    def test_fitting_tensor_generator_multi_input(self):
        train_generator = some_data_tensor_generator_multi_input(ModelMultiInputTest.batch_size)
        valid_generator = some_data_tensor_generator_multi_input(ModelMultiInputTest.batch_size)
        logs = self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelMultiInputTest.epochs,
            steps_per_epoch=ModelMultiInputTest.steps_per_epoch,
            validation_steps=ModelMultiInputTest.steps_per_epoch,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelMultiInputTest.epochs,
            'steps': ModelMultiInputTest.steps_per_epoch,
            'valid_steps': ModelMultiInputTest.steps_per_epoch,
        }
        self._test_callbacks_train(params, logs)

    def test_fitting_with_tensor_multi_input(self):
        train_real_steps_per_epoch = 30
        train_batch_size = ModelMultiInputTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples
        train_x = (torch.rand(train_size, 1), torch.rand(train_size, 1))
        train_y = torch.rand(train_size, 1)

        valid_real_steps_per_epoch = 10
        # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_batch_size = train_batch_size
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples
        valid_x = (torch.rand(valid_size, 1), torch.rand(valid_size, 1))
        valid_y = torch.rand(valid_size, 1)

        logs = self.model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=ModelMultiInputTest.epochs,
            batch_size=train_batch_size,
            steps_per_epoch=None,
            validation_steps=None,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelMultiInputTest.epochs,
            'steps': train_real_steps_per_epoch,
            'valid_steps': valid_real_steps_per_epoch,
        }
        self._test_callbacks_train(params, logs)

    def test_tensor_train_on_batch_multi_input(self):
        x1 = torch.rand(ModelMultiInputTest.batch_size, 1)
        x2 = torch.rand(ModelMultiInputTest.batch_size, 1)
        y = torch.rand(ModelMultiInputTest.batch_size, 1)
        loss = self.model.train_on_batch((x1, x2), y)
        self.assertEqual(type(loss), float)

    def test_train_on_batch_with_pred_multi_input(self):
        x1 = torch.rand(ModelMultiInputTest.batch_size, 1)
        x2 = torch.rand(ModelMultiInputTest.batch_size, 1)
        y = torch.rand(ModelMultiInputTest.batch_size, 1)
        loss, pred_y = self.model.train_on_batch((x1, x2), y, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(pred_y.shape, (ModelMultiInputTest.batch_size, 1))

    def test_ndarray_train_on_batch_multi_input(self):
        x1 = np.random.rand(ModelMultiInputTest.batch_size, 1).astype(np.float32)
        x2 = np.random.rand(ModelMultiInputTest.batch_size, 1).astype(np.float32)
        y = np.random.rand(ModelMultiInputTest.batch_size, 1).astype(np.float32)
        loss = self.model.train_on_batch((x1, x2), y)
        self.assertEqual(type(loss), float)

    def test_evaluate_multi_input(self):
        x = (
            torch.rand(ModelMultiInputTest.evaluate_dataset_len, 1),
            torch.rand(ModelMultiInputTest.evaluate_dataset_len, 1),
        )
        y = torch.rand(ModelMultiInputTest.evaluate_dataset_len, 1)
        loss = self.model.evaluate(x, y, batch_size=ModelMultiInputTest.batch_size)
        self.assertEqual(type(loss), float)

    def test_evaluate_with_pred_multi_input(self):
        x = (
            torch.rand(ModelMultiInputTest.evaluate_dataset_len, 1),
            torch.rand(ModelMultiInputTest.evaluate_dataset_len, 1),
        )
        y = torch.rand(ModelMultiInputTest.evaluate_dataset_len, 1)
        # We also test the unpacking.
        _, pred_y = self.model.evaluate(x, y, batch_size=ModelMultiInputTest.batch_size, return_pred=True)
        self.assertEqual(pred_y.shape, (ModelMultiInputTest.evaluate_dataset_len, 1))

    def test_evaluate_with_np_array_multi_input(self):
        x1 = np.random.rand(ModelMultiInputTest.evaluate_dataset_len, 1).astype(np.float32)
        x2 = np.random.rand(ModelMultiInputTest.evaluate_dataset_len, 1).astype(np.float32)
        x = (x1, x2)
        y = np.random.rand(ModelMultiInputTest.evaluate_dataset_len, 1).astype(np.float32)
        loss, pred_y = self.model.evaluate(x, y, batch_size=ModelMultiInputTest.batch_size, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(pred_y.shape, (ModelMultiInputTest.evaluate_dataset_len, 1))

    def test_evaluate_data_loader_multi_input(self):
        x1 = torch.rand(ModelMultiInputTest.evaluate_dataset_len, 1)
        x2 = torch.rand(ModelMultiInputTest.evaluate_dataset_len, 1)
        y = torch.rand(ModelMultiInputTest.evaluate_dataset_len, 1)
        dataset = TensorDataset((x1, x2), y)
        generator = DataLoader(dataset, ModelMultiInputTest.batch_size)
        loss, pred_y = self.model.evaluate_generator(generator, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(pred_y.shape, (ModelMultiInputTest.evaluate_dataset_len, 1))

    def test_evaluate_generator_multi_input(self):
        num_steps = 10
        generator = some_data_tensor_generator_multi_input(ModelMultiInputTest.batch_size)
        loss, pred_y = self.model.evaluate_generator(generator, steps=num_steps, return_pred=True)
        self.assertEqual(type(loss), float)
        self.assertEqual(pred_y.shape, (num_steps * ModelMultiInputTest.batch_size, 1))

    def test_tensor_evaluate_on_batch_multi_input(self):
        x1 = torch.rand(ModelMultiInputTest.batch_size, 1)
        x2 = torch.rand(ModelMultiInputTest.batch_size, 1)
        y = torch.rand(ModelMultiInputTest.batch_size, 1)
        loss = self.model.evaluate_on_batch((x1, x2), y)
        self.assertEqual(type(loss), float)

    def test_predict_multi_input(self):
        x = (
            torch.rand(ModelMultiInputTest.evaluate_dataset_len, 1),
            torch.rand(ModelMultiInputTest.evaluate_dataset_len, 1),
        )
        pred_y = self.model.predict(x, batch_size=ModelMultiInputTest.batch_size)
        self.assertEqual(pred_y.shape, (ModelMultiInputTest.evaluate_dataset_len, 1))

    def test_predict_with_np_array_multi_input(self):
        x1 = np.random.rand(ModelMultiInputTest.evaluate_dataset_len, 1).astype(np.float32)
        x2 = np.random.rand(ModelMultiInputTest.evaluate_dataset_len, 1).astype(np.float32)
        x = (x1, x2)
        pred_y = self.model.predict(x, batch_size=ModelMultiInputTest.batch_size)
        self.assertEqual(pred_y.shape, (ModelMultiInputTest.evaluate_dataset_len, 1))

    def test_predict_generator_multi_input(self):
        num_steps = 10
        generator = some_data_tensor_generator_multi_input(ModelMultiInputTest.batch_size)
        generator = (x for x, _ in generator)
        pred_y = self.model.predict_generator(generator, steps=num_steps)
        self.assertEqual(type(pred_y), np.ndarray)
        self.assertEqual(pred_y.shape, (num_steps * ModelMultiInputTest.batch_size, 1))

    def test_tensor_predict_on_batch_multi_input(self):
        x1 = torch.rand(ModelMultiInputTest.batch_size, 1)
        x2 = torch.rand(ModelMultiInputTest.batch_size, 1)
        pred_y = self.model.predict_on_batch((x1, x2))
        self.assertEqual(pred_y.shape, (ModelMultiInputTest.batch_size, 1))
