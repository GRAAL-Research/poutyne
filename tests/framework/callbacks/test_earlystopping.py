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

import unittest

import torch
import torch.nn as nn

from poutyne import Model, EarlyStopping
from tests.framework.base import CaptureOutputBase
from tests.framework.tools import some_data_generator


class EarlyStoppingTest(CaptureOutputBase):
    batch_size = 20

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

    def test_integration(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=False)
        self.model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[earlystopper])

    def test_early_stopping_patience_of_1(self):
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=False)

        val_losses = [8, 4, 5, 2]
        early_stop_epoch = 3
        self._test_early_stopping(earlystopper, val_losses, early_stop_epoch)

    def test_early_stopping_with_delta(self):
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=3, patience=2, verbose=False)

        val_losses = [8, 4, 5, 2, 2]
        early_stop_epoch = 4
        self._test_early_stopping(earlystopper, val_losses, early_stop_epoch)

    def test_early_stopping_with_max(self):
        earlystopper = EarlyStopping(monitor='val_loss', mode='max', min_delta=0, patience=2, verbose=False)

        val_losses = [2, 8, 4, 5, 2]
        early_stop_epoch = 4
        self._test_early_stopping(earlystopper, val_losses, early_stop_epoch)

    def test_mode_not_min_max_raise_error(self):
        with self.assertRaises(ValueError):
            invalid_mode = "a_mode"
            EarlyStopping(monitor='val_loss', mode=invalid_mode, min_delta=0, patience=2, verbose=False)

    def test_early_stopping_with_verbose(self):
        earlystopper = EarlyStopping(monitor='val_loss', mode='max', min_delta=0, patience=2, verbose=True)

        val_losses = [2, 8, 4, 5, 2]
        early_stop_epoch = 4

        self._capture_output()

        self._test_early_stopping(earlystopper, val_losses, early_stop_epoch)

        self.assertStdoutContains(['Epoch 4: early stopping'])

    def _test_early_stopping(self, earlystopper, val_losses, early_stop_epoch):
        generator = some_data_generator(EarlyStoppingTest.batch_size)

        self.model.stop_training = False

        earlystopper.set_params({'epochs': len(val_losses), 'steps': 1})
        earlystopper.set_model(self.model)
        earlystopper.on_train_begin({})
        for epoch, val_loss in enumerate(val_losses, 1):
            earlystopper.on_epoch_begin(epoch, {})
            earlystopper.on_train_batch_begin(1, {})
            loss = self._update_model(generator)
            earlystopper.on_train_batch_end(1, {'batch': 1, 'size': EarlyStoppingTest.batch_size, 'loss': loss})
            earlystopper.on_epoch_end(epoch, {'epoch': epoch, 'loss': loss, 'val_loss': val_loss})
            self.assertEqual(self.model.stop_training, epoch == early_stop_epoch)
            if epoch == early_stop_epoch:
                break

        earlystopper.on_train_end({})

    def _update_model(self, generator):
        self.pytorch_network.zero_grad()

        x, y = next(generator)
        pred_y = self.pytorch_network(x)
        loss = self.loss_function(pred_y, y)
        loss.backward()

        self.optimizer.step()

        return float(loss)


if __name__ == '__main__':
    unittest.main()
