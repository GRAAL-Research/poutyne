import unittest
from unittest import TestCase

import torch
import torch.nn as nn

from pytoune import torch_to_numpy
from pytoune.framework import Model
from pytoune.framework.callbacks import BestModelRestore


def some_data_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y

class BestModelRestoreTest(TestCase):
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
        model_restore = BestModelRestore(monitor='val_loss', verbose=True)
        self.model.fit_generator(train_gen, valid_gen,
                                 epochs=10,
                                 steps_per_epoch=5,
                                 callbacks=[model_restore])

    def test_basic_restore(self):
        model_restore = BestModelRestore(monitor='val_loss')

        val_losses = [3, 2, 8, 5, 4]
        best_epoch = 2
        self._test_restore_with_val_losses(model_restore, val_losses, best_epoch)

    def test_save_best_only_with_max(self):
        model_restore = BestModelRestore(monitor='val_loss', mode='max')

        val_losses = [3, 2, 8, 5, 4]
        best_epoch = 3
        self._test_restore_with_val_losses(model_restore, val_losses, best_epoch)

    def _test_restore_with_val_losses(self, checkpointer, val_losses, best_epoch):
        generator = some_data_generator(BestModelRestoreTest.batch_size)

        best_epoch_weights = None
        checkpointer.set_params({'epochs': len(val_losses), 'steps': 1})
        checkpointer.set_model(self.model)
        checkpointer.on_train_begin({})
        for epoch, val_loss in enumerate(val_losses, 1):
            checkpointer.on_epoch_begin(epoch, {})
            checkpointer.on_batch_begin(1, {})
            loss = self._update_model(generator)
            checkpointer.on_batch_end(
                1, {'batch': 1, 'size': BestModelRestoreTest.batch_size, 'loss': loss}
            )
            checkpointer.on_epoch_end(epoch, {'epoch': epoch, 'loss': loss, 'val_loss': val_loss})
            if epoch == best_epoch:
                best_epoch_weights = torch_to_numpy(self.model.get_weight_copies())
        checkpointer.on_train_end({})

        final_weights = torch_to_numpy(self.model.get_weight_copies())
        self.assertEqual(best_epoch_weights, final_weights)

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
