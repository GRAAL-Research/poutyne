import os

import csv

from unittest import TestCase

from tempfile import TemporaryDirectory

from pytoune.framework import Model
from pytoune.framework.callbacks import CSVLogger, Callback

import torch
import torch.nn as nn

def some_data_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y

class History(Callback):
    def on_epoch_end(self, epoch, logs):
        self.history.append(logs)

    def on_batch_end(self, batch, logs):
        self.history.append(logs)

    def on_train_begin(self, logs):
        self.history = []


class LoggerTest(TestCase):
    batch_size = 20
    lr = 1e-3

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_module = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_module.parameters(), lr=LoggerTest.lr)
        self.model = Model(self.pytorch_module, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        self.csv_filename = os.path.join(self.temp_dir_obj.name, 'my_log.csv')

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_logging(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = CSVLogger(self.csv_filename)
        history = self.model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[logger])
        self._test_logging(history)

    def test_logging_with_batch_granularity(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = CSVLogger(self.csv_filename, batch_granularity=True)
        history = History()
        self.model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[logger, history])
        self._test_logging(history.history)

    def test_logging_append(self):
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        logger = CSVLogger(self.csv_filename)
        history = self.model.fit_generator(train_gen, valid_gen, epochs=10, steps_per_epoch=5, callbacks=[logger])
        logger = CSVLogger(self.csv_filename, append=True)
        history2 = self.model.fit_generator(train_gen, valid_gen, epochs=20, steps_per_epoch=5, initial_epoch=10, callbacks=[logger])
        self._test_logging(history + history2)

    def _test_logging(self, history):
        with open(self.csv_filename) as csvfile:
            reader = csv.DictReader(csvfile)
            rows = []
            for row in reader:
                if row['epoch'] != '':
                    self.assertAlmostEqual(float(row['lr']), LoggerTest.lr)
                del row['lr']
                rows.append(row)
        self.assertEqual(len(rows), len(history))
        for row, hist_entry in zip(rows, history):
            row = {k:v for k, v in row.items() if v != ''}
            self.assertEqual(row.keys(), hist_entry.keys())
            for k in row.keys():
                if isinstance(hist_entry[k], float):
                    self.assertAlmostEqual(float(row[k]), hist_entry[k])
                else:
                    self.assertEqual(str(row[k]), str(hist_entry[k]))



if __name__ == '__main__':
    unittest.main()
