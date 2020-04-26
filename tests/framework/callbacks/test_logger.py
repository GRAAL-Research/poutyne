import csv
import os
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf, main
from unittest.mock import MagicMock, call

import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
except ImportError:
    TorchSummaryWriter = None

try:
    from tensorboardX import SummaryWriter as XSummaryWriter
except ImportError:
    XSummaryWriter = None

from poutyne.framework import Model, Callback, TensorBoardLogger
from poutyne.framework import CSVLogger as NonAtomicCSVLogger
from poutyne.framework import AtomicCSVLogger


def some_data_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y


class History(Callback):
    def on_epoch_end(self, epoch_number, logs):
        self.history.append(logs)

    def on_train_batch_end(self, batch_number, logs):
        self.history.append(logs)

    def on_train_begin(self, logs):
        self.history = []


class BaseCSVLoggerTest:
    # pylint: disable=not-callable,no-member
    CSVLogger = None
    batch_size = 20
    lr = 1e-3
    num_epochs = 10

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=BaseCSVLoggerTest.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        self.csv_filename = os.path.join(self.temp_dir_obj.name, 'my_log.csv')

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_logging(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = self.CSVLogger(self.csv_filename)
        history = self.model.fit_generator(train_gen,
                                           valid_gen,
                                           epochs=self.num_epochs,
                                           steps_per_epoch=5,
                                           callbacks=[logger])
        self._test_logging(history)

    def test_logging_with_batch_granularity(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = self.CSVLogger(self.csv_filename, batch_granularity=True)
        history = History()
        self.model.fit_generator(train_gen,
                                 valid_gen,
                                 epochs=self.num_epochs,
                                 steps_per_epoch=5,
                                 callbacks=[logger, history])
        self._test_logging(history.history)

    def test_logging_append(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = self.CSVLogger(self.csv_filename)
        history = self.model.fit_generator(train_gen,
                                           valid_gen,
                                           epochs=self.num_epochs,
                                           steps_per_epoch=5,
                                           callbacks=[logger])
        logger = self.CSVLogger(self.csv_filename, append=True)
        history2 = self.model.fit_generator(train_gen,
                                            valid_gen,
                                            epochs=20,
                                            steps_per_epoch=5,
                                            initial_epoch=self.num_epochs,
                                            callbacks=[logger])
        self._test_logging(history + history2)

    def test_logging_overwrite(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = self.CSVLogger(self.csv_filename)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger])
        logger = self.CSVLogger(self.csv_filename, append=False)
        history = self.model.fit_generator(train_gen,
                                           valid_gen,
                                           epochs=20,
                                           steps_per_epoch=5,
                                           initial_epoch=self.num_epochs,
                                           callbacks=[logger])
        self._test_logging(history)

    def _test_logging(self, history):
        with open(self.csv_filename) as csvfile:
            reader = csv.DictReader(csvfile)
            rows = []
            for row in reader:
                if row['epoch'] != '':
                    self.assertAlmostEqual(float(row['lr']), BaseCSVLoggerTest.lr)
                del row['lr']
                rows.append(row)
        self.assertEqual(len(rows), len(history))
        for row, hist_entry in zip(rows, history):
            row = {k: v for k, v in row.items() if v != ''}
            self.assertEqual(row.keys(), hist_entry.keys())
            for k in row.keys():
                if isinstance(hist_entry[k], float):
                    self.assertAlmostEqual(float(row[k]), hist_entry[k])
                else:
                    self.assertEqual(str(row[k]), str(hist_entry[k]))


class NonAtomicCSVLoggerTest(BaseCSVLoggerTest, TestCase):
    CSVLogger = NonAtomicCSVLogger


class AtomicCSVLoggerTest(BaseCSVLoggerTest, TestCase):
    CSVLogger = AtomicCSVLogger


class BaseTensorBoardLoggerTest:
    SummaryWriter = None
    batch_size = 20
    lr = 1e-3
    num_epochs = 10

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=BaseTensorBoardLoggerTest.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        # pylint: disable=not-callable
        self.writer = self.SummaryWriter(self.temp_dir_obj.name)
        self.writer.add_scalars = MagicMock()

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_logging(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = TensorBoardLogger(self.writer)
        history = self.model.fit_generator(train_gen,
                                           valid_gen,
                                           epochs=self.num_epochs,
                                           steps_per_epoch=5,
                                           callbacks=[logger])
        self._test_logging(history)

    def _test_logging(self, history):
        calls = list()
        for h in history:
            calls.append(call('loss', {'loss': h['loss'], 'val_loss': h['val_loss']}, h['epoch']))
            calls.append(call('lr', {'lr': self.lr}, h['epoch']))
        self.writer.add_scalars.assert_has_calls(calls, any_order=True)


@skipIf(XSummaryWriter is None, "Needs tensorboardX to run this test")
class TensorboardXLoggerTest(BaseTensorBoardLoggerTest, TestCase):
    SummaryWriter = XSummaryWriter


@skipIf(TorchSummaryWriter is None, "Unable to import SummaryWriter from torch")
class TorchTensorboardLoggerTest(BaseTensorBoardLoggerTest, TestCase):
    SummaryWriter = TorchSummaryWriter


if __name__ == '__main__':
    main()
