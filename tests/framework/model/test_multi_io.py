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

from poutyne import Model

from .base import ModelFittingTestCase, MultiIOModel


def some_data_tensor_generator_multi_io(batch_size):
    while True:
        x1 = torch.rand(batch_size, 1)
        x2 = torch.rand(batch_size, 1)
        y1 = torch.rand(batch_size, 1)
        y2 = torch.rand(batch_size, 1)
        yield (x1, x2), (y1, y2)


class ModelMultiIOTest(ModelFittingTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.pytorch_network = MultiIOModel(num_input=2, num_output=2)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)

        self.model = Model(
            self.pytorch_network,
            self.optimizer,
            lambda y_pred, y_true: self.loss_function(y_pred[0], y_true[0]) + self.loss_function(y_pred[1], y_true[1]),
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics,
        )

    def test_fitting_tensor_generator_multi_io(self):
        train_generator = some_data_tensor_generator_multi_io(ModelMultiIOTest.batch_size)
        valid_generator = some_data_tensor_generator_multi_io(ModelMultiIOTest.batch_size)
        logs = self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelMultiIOTest.epochs,
            steps_per_epoch=ModelMultiIOTest.steps_per_epoch,
            validation_steps=ModelMultiIOTest.steps_per_epoch,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelMultiIOTest.epochs,
            'steps': ModelMultiIOTest.steps_per_epoch,
            'valid_steps': ModelMultiIOTest.steps_per_epoch,
        }
        self._test_callbacks_train(params, logs)

    def test_fitting_with_tensor_multi_io(self):
        train_real_steps_per_epoch = 30
        train_batch_size = ModelMultiIOTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples
        train_x = (torch.rand(train_size, 1), torch.rand(train_size, 1))
        train_y = (torch.rand(train_size, 1), torch.rand(train_size, 1))

        valid_real_steps_per_epoch = 10
        # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_batch_size = train_batch_size
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples
        valid_x = (torch.rand(valid_size, 1), torch.rand(valid_size, 1))
        valid_y = (torch.rand(valid_size, 1), torch.rand(valid_size, 1))

        logs = self.model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=ModelMultiIOTest.epochs,
            batch_size=train_batch_size,
            steps_per_epoch=None,
            validation_steps=None,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelMultiIOTest.epochs,
            'steps': train_real_steps_per_epoch,
            'valid_steps': valid_real_steps_per_epoch,
        }
        self._test_callbacks_train(params, logs)

    def test_tensor_train_on_batch_multi_io(self):
        x1 = torch.rand(ModelMultiIOTest.batch_size, 1)
        x2 = torch.rand(ModelMultiIOTest.batch_size, 1)
        y1 = torch.rand(ModelMultiIOTest.batch_size, 1)
        y2 = torch.rand(ModelMultiIOTest.batch_size, 1)
        loss = self.model.train_on_batch((x1, x2), (y1, y2))
        self.assertEqual(type(loss), float)

    def test_ndarray_train_on_batch_multi_io(self):
        x1 = np.random.rand(ModelMultiIOTest.batch_size, 1).astype(np.float32)
        x2 = np.random.rand(ModelMultiIOTest.batch_size, 1).astype(np.float32)
        y1 = np.random.rand(ModelMultiIOTest.batch_size, 1).astype(np.float32)
        y2 = np.random.rand(ModelMultiIOTest.batch_size, 1).astype(np.float32)
        loss = self.model.train_on_batch((x1, x2), (y1, y2))
        self.assertEqual(type(loss), float)

    def test_evaluate_with_pred_multi_io(self):
        x = (torch.rand(ModelMultiIOTest.evaluate_dataset_len, 1), torch.rand(ModelMultiIOTest.evaluate_dataset_len, 1))
        y = (torch.rand(ModelMultiIOTest.evaluate_dataset_len, 1), torch.rand(ModelMultiIOTest.evaluate_dataset_len, 1))
        # We also test the unpacking.
        _, pred_y = self.model.evaluate(x, y, batch_size=ModelMultiIOTest.batch_size, return_pred=True)
        for pred in pred_y:
            self.assertEqual(pred.shape, (ModelMultiIOTest.evaluate_dataset_len, 1))

    def test_tensor_evaluate_on_batch_multi_io(self):
        y = (torch.rand(ModelMultiIOTest.batch_size, 1), torch.rand(ModelMultiIOTest.batch_size, 1))
        x = (torch.rand(ModelMultiIOTest.batch_size, 1), torch.rand(ModelMultiIOTest.batch_size, 1))
        loss = self.model.evaluate_on_batch(x, y)
        self.assertEqual(type(loss), float)

    def test_predict_with_np_array_multi_io(self):
        x1 = np.random.rand(ModelMultiIOTest.evaluate_dataset_len, 1).astype(np.float32)
        x2 = np.random.rand(ModelMultiIOTest.evaluate_dataset_len, 1).astype(np.float32)
        x = (x1, x2)
        pred_y = self.model.predict(x, batch_size=ModelMultiIOTest.batch_size)
        for pred in pred_y:
            self.assertEqual(pred.shape, (ModelMultiIOTest.evaluate_dataset_len, 1))

    def test_predict_generator_multi_io(self):
        num_steps = 10
        generator = some_data_tensor_generator_multi_io(ModelMultiIOTest.batch_size)
        generator = (x for x, _ in generator)
        pred_y = self.model.predict_generator(generator, steps=num_steps)

        for pred in pred_y:
            self.assertEqual(pred.shape, (num_steps * ModelMultiIOTest.batch_size, 1))

    def test_tensor_predict_on_batch_multi_io(self):
        x = (torch.rand(ModelMultiIOTest.batch_size, 1), torch.rand(ModelMultiIOTest.batch_size, 1))
        pred_y = self.model.predict_on_batch(x)
        self._test_size_and_type_for_generator(pred_y, (ModelMultiIOTest.batch_size, 1))
