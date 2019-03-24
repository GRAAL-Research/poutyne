import unittest
from unittest import TestCase

import torch
import torch.nn as nn

from poutyne.framework import Model
from poutyne.framework.callbacks import EarlyStopping


def some_data_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y

class EarlyStoppingTest(TestCase):
    batch_size = 20

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_module = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_module.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function)

    def test_integration(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=False)
        self.model.fit_generator(train_gen, valid_gen,
                                 epochs=10,
                                 steps_per_epoch=5,
                                 callbacks=[earlystopper])

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
        earlystopper = EarlyStopping(monitor='val_loss',
                                     mode='max',
                                     min_delta=0,
                                     patience=2,
                                     verbose=False)

        val_losses = [2, 8, 4, 5, 2]
        early_stop_epoch = 4
        self._test_early_stopping(earlystopper, val_losses, early_stop_epoch)

    def _test_early_stopping(self, earlystopper, val_losses, early_stop_epoch):
        generator = some_data_generator(EarlyStoppingTest.batch_size)

        self.model.stop_training = False

        earlystopper.set_params({'epochs': len(val_losses), 'steps': 1})
        earlystopper.set_model(self.model)
        earlystopper.on_train_begin({})
        for epoch, val_loss in enumerate(val_losses, 1):
            earlystopper.on_epoch_begin(epoch, {})
            earlystopper.on_batch_begin(1, {})
            loss = self._update_model(generator)
            earlystopper.on_batch_end(
                1, {'batch': 1, 'size': EarlyStoppingTest.batch_size, 'loss': loss}
            )
            earlystopper.on_epoch_end(epoch, {'epoch': epoch, 'loss': loss, 'val_loss': val_loss})
            self.assertEqual(self.model.stop_training, epoch == early_stop_epoch)
            if epoch == early_stop_epoch:
                break

        earlystopper.on_train_end({})

    def _update_model(self, generator):
        self.pytorch_module.zero_grad()

        x, y = next(generator)
        pred_y = self.pytorch_module(x)
        loss = self.loss_function(pred_y, y)
        loss.backward()

        self.optimizer.step()

        return float(loss)

if __name__ == '__main__':
    unittest.main()
