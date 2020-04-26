import csv
import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
import torch.nn as nn

from poutyne.framework import Model
from poutyne.framework import CSVGradientLogger as NonAtomicCSVGradientLogger
from poutyne.framework import AtomicCSVGradientLogger, MemoryGradientLogger


def some_data_generator(batch_size):
    while True:
        x = torch.rand(batch_size, 1)
        y = torch.rand(batch_size, 1)
        yield x, y


class BaseCSVGradientLoggerTest:
    # pylint: disable=not-callable,no-member
    CSVGradientLogger = None
    batch_size = 20
    num_epochs = 10

    def setUp(self):
        torch.manual_seed(42)
        self.pytorch_network = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 1))
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)
        self.temp_dir_obj = TemporaryDirectory()
        self.csv_filename = os.path.join(self.temp_dir_obj.name, 'layer_{}.csv')

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_logging(self):
        train_gen = some_data_generator(self.batch_size)
        valid_gen = some_data_generator(self.batch_size)
        memgrad = MemoryGradientLogger()
        logger = self.CSVGradientLogger(self.csv_filename)
        self.model.fit_generator(train_gen,
                                 valid_gen,
                                 epochs=self.num_epochs,
                                 steps_per_epoch=5,
                                 callbacks=[memgrad, logger])
        self._test_logging(memgrad.history)

    def test_logging_append(self):
        train_gen = some_data_generator(self.batch_size)
        valid_gen = some_data_generator(self.batch_size)
        logger = self.CSVGradientLogger(self.csv_filename)
        memgrad = MemoryGradientLogger()
        self.model.fit_generator(train_gen,
                                 valid_gen,
                                 epochs=self.num_epochs,
                                 steps_per_epoch=5,
                                 callbacks=[memgrad, logger])
        memgrad2 = MemoryGradientLogger()
        logger = self.CSVGradientLogger(self.csv_filename, append=True)
        self.model.fit_generator(train_gen,
                                 valid_gen,
                                 epochs=20,
                                 steps_per_epoch=5,
                                 initial_epoch=self.num_epochs,
                                 callbacks=[memgrad2, logger])
        history = {layer: stats1 + memgrad2.history[layer] for layer, stats1 in memgrad.history.items()}
        self._test_logging(history)

    def test_logging_overwrite(self):
        train_gen = some_data_generator(self.batch_size)
        valid_gen = some_data_generator(self.batch_size)
        logger = self.CSVGradientLogger(self.csv_filename)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[logger])
        memgrad = MemoryGradientLogger()
        logger = self.CSVGradientLogger(self.csv_filename, append=False)
        self.model.fit_generator(train_gen,
                                 valid_gen,
                                 epochs=20,
                                 steps_per_epoch=5,
                                 initial_epoch=self.num_epochs,
                                 callbacks=[memgrad, logger])
        self._test_logging(memgrad.history)

    def _test_logging(self, history):
        for layer, stats in history.items():
            filename = self.csv_filename.format(layer)
            with open(filename) as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
            self.assertEqual(len(rows), len(stats))
            for row, stats_entry in zip(rows, stats):
                self.assertEqual(row.keys(), stats_entry.keys())
                for k in row.keys():
                    self.assertAlmostEqual(float(row[k]), stats_entry[k])


class NonAtomicCSVGradientLoggerTest(BaseCSVGradientLoggerTest, TestCase):
    CSVGradientLogger = NonAtomicCSVGradientLogger


class AtomicCSVGradientLoggerTest(BaseCSVGradientLoggerTest, TestCase):
    CSVGradientLogger = AtomicCSVGradientLogger
