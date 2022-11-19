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

from .base import ModelFittingTestCase


class DictOutputModel(nn.Module):
    """Model to test multiple dictionnary output"""

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(1, 1)
        self.output1 = nn.Linear(1, 1)
        self.output2 = nn.Linear(1, 1)

    def forward(self, x):
        out1 = self.output1(self.input(x))
        out2 = self.output2(self.input(x))
        return {'out1': out1, 'out2': out2}


class ModelDictOutputTest(ModelFittingTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.pytorch_network = DictOutputModel()
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)

        self.model = Model(
            self.pytorch_network,
            self.optimizer,
            lambda y_p, y_t: self.loss_function(y_p['out1'], y_t[0]) + self.loss_function(y_p['out2'], y_t[1]),
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics,
        )

    def test_fitting_with_tensor_multi_output_dict(self):
        train_real_steps_per_epoch = 30
        train_batch_size = ModelDictOutputTest.batch_size
        train_final_batch_missing_samples = 7
        train_size = train_real_steps_per_epoch * train_batch_size - train_final_batch_missing_samples
        train_x = torch.rand(train_size, 1)
        train_y = (torch.rand(train_size, 1), torch.rand(train_size, 1))

        valid_real_steps_per_epoch = 10
        # valid_batch_size will be the same as train_batch_size in the fit method.
        valid_batch_size = train_batch_size
        valid_final_batch_missing_samples = 3
        valid_size = valid_real_steps_per_epoch * valid_batch_size - valid_final_batch_missing_samples
        valid_x = torch.rand(valid_size, 1)
        valid_y = (torch.rand(valid_size, 1), torch.rand(valid_size, 1))

        logs = self.model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=ModelDictOutputTest.epochs,
            batch_size=train_batch_size,
            steps_per_epoch=None,
            validation_steps=None,
            callbacks=[self.mock_callback],
        )
        params = {
            'epochs': ModelDictOutputTest.epochs,
            'steps': train_real_steps_per_epoch,
            'valid_steps': valid_real_steps_per_epoch,
        }
        self._test_callbacks_train(params, logs)

    def test_ndarray_train_on_batch_dict_output(self):
        x = np.random.rand(ModelDictOutputTest.batch_size, 1).astype(np.float32)
        y1 = np.random.rand(ModelDictOutputTest.batch_size, 1).astype(np.float32)
        y2 = np.random.rand(ModelDictOutputTest.batch_size, 1).astype(np.float32)
        loss = self.model.train_on_batch(x, (y1, y2))
        self.assertEqual(type(loss), float)

    def test_evaluate_with_pred_dict_output(self):
        y = (
            torch.rand(ModelDictOutputTest.evaluate_dataset_len, 1),
            torch.rand(ModelDictOutputTest.evaluate_dataset_len, 1),
        )
        x = torch.rand(ModelDictOutputTest.evaluate_dataset_len, 1)
        # We also test the unpacking.
        _, pred_y = self.model.evaluate(x, y, batch_size=ModelDictOutputTest.batch_size, return_pred=True)
        for pred in pred_y.values():
            self.assertEqual(pred.shape, (ModelDictOutputTest.evaluate_dataset_len, 1))
