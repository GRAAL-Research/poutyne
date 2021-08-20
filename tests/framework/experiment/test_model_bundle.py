import os
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch, MagicMock

import torch
from pandas import DataFrame, Series
from torch import nn
from torch.utils.data import TensorDataset

from poutyne import ModelBundle
from tests.framework.tools import SomeDataGeneratorWithLen


class ModelBundleTest(TestCase):
    NUM_EPOCHS = 5

    def setUp(self):
        self.temp_dir_obj = TemporaryDirectory()
        self.test_checkpoints_path = os.path.join(self.temp_dir_obj.name, 'expt')

        self.test_experiment = ModelBundle.from_network(self.test_checkpoints_path,
                                                        nn.Linear(1, 1),
                                                        optimizer='sgd',
                                                        loss_function='mse',
                                                        monitor_metric="loss",
                                                        monitor_mode="min")
        self.ckpt_1_path = os.path.join(self.test_checkpoints_path, "checkpoint_epoch_1.ckpt")
        self.ckpt_last_path = os.path.join(self.test_checkpoints_path, "checkpoint.ckpt")
        self.optim_ckpt_path = os.path.join(self.test_checkpoints_path, "checkpoint.optim")
        self.tsv_log_path = os.path.join(self.test_checkpoints_path, "log.tsv")
        self.tsv_test_log_path = os.path.join(self.test_checkpoints_path, "test_log.tsv")
        self.epoch_file_path = os.path.join(self.test_checkpoints_path, "last.epoch")

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_integration_train(self):
        train_generator = SomeDataGeneratorWithLen(32, 10, 0)
        valid_generator = SomeDataGeneratorWithLen(32, 10, 0)
        logs = self.test_experiment.train(train_generator, valid_generator, epochs=ModelBundleTest.NUM_EPOCHS)
        self._test_train_integration(logs)

    def test_integration_train_dataset(self):
        train_dataset = TensorDataset(torch.rand(32 * 10, 1), torch.rand(32 * 10, 1))
        valid_dataset = TensorDataset(torch.rand(32 * 10, 1), torch.rand(32 * 10, 1))
        logs = self.test_experiment.train_dataset(train_dataset, valid_dataset, epochs=ModelBundleTest.NUM_EPOCHS)
        self._test_train_integration(logs)

    def test_integration_train_data(self):
        train_x, train_y = torch.rand(32 * 10, 1), torch.rand(32 * 10, 1)
        validation_data = torch.rand(32 * 10, 1), torch.rand(32 * 10, 1)
        logs = self.test_experiment.train_data(train_x, train_y, validation_data, epochs=ModelBundleTest.NUM_EPOCHS)
        self._test_train_integration(logs)

    def test_train_resume(self):
        train_generator = SomeDataGeneratorWithLen(32, 10, 0)
        valid_generator = SomeDataGeneratorWithLen(32, 10, 0)
        logs = self.test_experiment.train(train_generator, valid_generator, epochs=ModelBundleTest.NUM_EPOCHS)
        self._test_train_integration(logs)

        epochs = ModelBundleTest.NUM_EPOCHS + 10
        logs = self.test_experiment.train(train_generator, valid_generator, epochs=epochs)
        self._test_train_integration(logs, epochs=epochs, initial_epoch=ModelBundleTest.NUM_EPOCHS + 1)

    def test_train_no_log(self):
        test_experiment = ModelBundle.from_network(self.test_checkpoints_path,
                                                   nn.Linear(1, 1),
                                                   optimizer='sgd',
                                                   loss_function='mse',
                                                   monitor_metric="loss",
                                                   monitor_mode="min",
                                                   logging=False)
        train_generator = SomeDataGeneratorWithLen(32, 10, 0)
        valid_generator = SomeDataGeneratorWithLen(32, 10, 0)
        logs = test_experiment.train(train_generator, valid_generator, epochs=ModelBundleTest.NUM_EPOCHS)

        self.assertFalse(os.path.isdir(self.test_checkpoints_path))
        self.assertFalse(os.path.isfile(self.ckpt_1_path))
        self.assertFalse(os.path.isfile(self.ckpt_last_path))
        self.assertFalse(os.path.isfile(self.optim_ckpt_path))
        self.assertFalse(os.path.isfile(self.tsv_log_path))
        self.assertFalse(os.path.isfile(self.epoch_file_path))
        self.assertFalse(os.path.isfile(self.tsv_test_log_path))

        self.assertEqual(len(logs), ModelBundleTest.NUM_EPOCHS)
        for i, log in enumerate(logs, 1):
            self.assertIn('epoch', log)
            self.assertEqual(log['epoch'], i)
            self.assertIn('loss', log)
            self.assertIn('val_loss', log)
            self.assertIn('time', log)

    def _test_train_integration(self, logs, epochs=NUM_EPOCHS, initial_epoch=1):
        self.assertTrue(os.path.isdir(self.test_checkpoints_path))
        self.assertTrue(os.path.isfile(self.ckpt_1_path))
        self.assertTrue(os.path.isfile(self.ckpt_last_path))
        self.assertTrue(os.path.isfile(self.optim_ckpt_path))
        self.assertTrue(os.path.isfile(self.tsv_log_path))
        self.assertTrue(os.path.isfile(self.epoch_file_path))
        self.assertFalse(os.path.isfile(self.tsv_test_log_path))

        self.assertEqual(len(logs), epochs - initial_epoch + 1)
        for i, log in enumerate(logs, initial_epoch):
            self.assertIn('epoch', log)
            self.assertEqual(log['epoch'], i)
            self.assertIn('loss', log)
            self.assertIn('val_loss', log)
            self.assertIn('time', log)

        with open(self.epoch_file_path, 'r') as fd:
            epoch = int(fd.read())
        self.assertEqual(epoch, epochs)

    def test_integration_test(self):
        self._train_expt()
        generator = SomeDataGeneratorWithLen(32, 10, 0)
        logs = self.test_experiment.test(generator)
        self._test_test_integration(logs)

    def test_integration_test_dataset(self):
        self._train_expt()
        dataset = TensorDataset(torch.rand(32 * 10, 1), torch.rand(32 * 10, 1))
        logs = self.test_experiment.test_dataset(dataset)
        self._test_test_integration(logs)

    def test_integration_test_data(self):
        self._train_expt()
        x, y = torch.rand(32 * 10, 1), torch.rand(32 * 10, 1)
        log = self.test_experiment.test_data(x, y)
        self._test_test_integration(log)

    def _train_expt(self):
        train_generator = SomeDataGeneratorWithLen(32, 10, 0)
        valid_generator = SomeDataGeneratorWithLen(32, 10, 0)
        self.test_experiment.train(train_generator, valid_generator, epochs=ModelBundleTest.NUM_EPOCHS)

    def _test_test_integration(self, log):
        self.assertTrue(os.path.isfile(self.tsv_test_log_path))

        self.assertIn('test_loss', log)
        self.assertIn('time', log)

    def setUpTwoModelBundle(self, a_params=None, b_params=None):
        self.test_checkpoints_path_b = os.path.join(self.temp_dir_obj.name, 'expt_b')
        self.test_checkpoints_path_a = os.path.join(self.temp_dir_obj.name, 'expt_a')

        if a_params is None:
            a_params = {}

        self.test_experiment_a = ModelBundle.from_network(self.test_checkpoints_path_a, nn.Linear(1, 1), **a_params)

        if b_params is None:
            b_params = {}

        self.test_experiment_b = ModelBundle.from_network(self.test_checkpoints_path_b, nn.Linear(1, 1), **b_params)

    def test_givenAIsSmallerThanBMinMonitoring_thenReturnTrue(self):
        self.setUpTwoModelBundle()

        with patch.object(ModelBundle, "load_checkpoint") as load_checkpoint_mock:
            mocked_stats = MagicMock(spec=DataFrame)
            series_mock = MagicMock(spec=Series)
            series_mock.item.side_effect = [[1], [2]]  # The monitored metric values
            mocked_stats.__getitem__.return_value = series_mock
            load_checkpoint_mock.return_value = mocked_stats

            self.assertTrue(self.test_experiment_a.is_better_than(self.test_experiment_b))

    def test_givenAIsGreaterThanBMinMonitoring_thenReturnFalse(self):
        self.setUpTwoModelBundle()

        with patch.object(ModelBundle, "load_checkpoint") as load_checkpoint_mock:
            mocked_stats = MagicMock(spec=DataFrame)
            series_mock = MagicMock(spec=Series)
            series_mock.item.side_effect = [[2], [1]]  # The monitored metric values
            mocked_stats.__getitem__.return_value = series_mock
            load_checkpoint_mock.return_value = mocked_stats

            self.assertFalse(self.test_experiment_a.is_better_than(self.test_experiment_b))

    def test_givenAIsSmallerThanBMaxMonitoring_thenReturnFalse(self):
        # We need to specify the metric for the mode to be properly set to "max"
        params = {"monitor_mode": "max", "monitor_metric": "loss"}
        self.setUpTwoModelBundle(a_params=params, b_params=params)

        with patch.object(ModelBundle, "load_checkpoint") as load_checkpoint_mock:
            mocked_stats = MagicMock(spec=DataFrame)
            series_mock = MagicMock(spec=Series)
            series_mock.item.side_effect = [[1], [2]]  # The monitored metric values
            mocked_stats.__getitem__.return_value = series_mock
            load_checkpoint_mock.return_value = mocked_stats

            self.assertFalse(self.test_experiment_a.is_better_than(self.test_experiment_b))

    def test_givenAIsGreaterThanBMaxMonitoring_thenReturnTrue(self):
        # We need to specify the metric for the mode to be properly set to "max"
        params = {"monitor_mode": "max", "monitor_metric": "loss"}
        self.setUpTwoModelBundle(a_params=params, b_params=params)

        with patch.object(ModelBundle, "load_checkpoint") as load_checkpoint_mock:
            mocked_stats = MagicMock(spec=DataFrame)
            series_mock = MagicMock(spec=Series)
            series_mock.item.side_effect = [[2], [1]]  # The monitored metric values
            mocked_stats.__getitem__.return_value = series_mock
            load_checkpoint_mock.return_value = mocked_stats

            self.assertTrue(self.test_experiment_a.is_better_than(self.test_experiment_b))

    def test_givenSomeModelBundleNotLogging_thenRaiseValueError(self):
        params_a = {"logging": False}
        params_b = {"logging": True}
        self.setUpTwoModelBundle(a_params=params_a, b_params=params_b)
        with self.assertRaises(ValueError):
            self.test_experiment_a.is_better_than(self.test_experiment_b)

        params_a = {"logging": True}
        params_b = {"logging": False}
        self.setUpTwoModelBundle(a_params=params_a, b_params=params_b)
        with self.assertRaises(ValueError):
            self.test_experiment_a.is_better_than(self.test_experiment_b)

        params = {"logging": False}
        self.setUpTwoModelBundle(a_params=params, b_params=params)
        with self.assertRaises(ValueError):
            self.test_experiment_a.is_better_than(self.test_experiment_b)

    def test_givenDifferentMonitorMetric_thenRaiseValueError(self):
        params_a = {"monitor_metric": "loss"}
        params_b = {"monitor_metric": "acc"}
        self.setUpTwoModelBundle(a_params=params_a, b_params=params_b)
        with self.assertRaises(ValueError):
            self.test_experiment_a.is_better_than(self.test_experiment_b)

    def test_givenDifferentMonitorMode_thenRaiseValueError(self):
        # We need to specify the metric for the mode to be properly set to "max"
        params_a = {"monitor_mode": "max", "monitor_metric": "loss"}
        params_b = {"monitor_mode": "min"}
        self.setUpTwoModelBundle(a_params=params_a, b_params=params_b)
        with self.assertRaises(ValueError):
            self.test_experiment_a.is_better_than(self.test_experiment_b)


class ModelBundleCheckpointLoadingTest(TestCase):

    def setUp(self):
        self.temp_dir_obj = TemporaryDirectory()
        self.test_checkpoints_path = self.temp_dir_obj.name
        self.ckpt_1_path = os.path.join(self.test_checkpoints_path, "checkpoint_epoch_1.ckpt")
        self.ckpt_last_path = os.path.join(self.test_checkpoints_path, "checkpoint.ckpt")

        expt = ModelBundle.from_network(self.test_checkpoints_path,
                                        nn.Linear(1, 1),
                                        optimizer='sgd',
                                        loss_function='mse',
                                        monitor_metric="loss",
                                        monitor_mode="min")
        train_generator = SomeDataGeneratorWithLen(2, 32, 0)
        expt.train(train_generator, epochs=1)

        self.test_experiment = ModelBundle.from_network(self.test_checkpoints_path,
                                                        nn.Linear(1, 1),
                                                        optimizer='sgd',
                                                        loss_function='mse',
                                                        monitor_metric="loss",
                                                        monitor_mode="min")

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_load_checkpoint_with_int(self):
        self.test_experiment.load_checkpoint(1)

        self.assertEqual(self.test_experiment.model.network.state_dict(),
                         torch.load(self.ckpt_1_path, map_location="cpu"))

    def test_load_checkpoint_best(self):
        self.test_experiment.load_checkpoint("best")

        self.assertEqual(self.test_experiment.model.network.state_dict(),
                         torch.load(self.ckpt_1_path, map_location="cpu"))

    def test_load_checkpoint_last(self):
        self.test_experiment.load_checkpoint("last")

        self.assertEqual(self.test_experiment.model.network.state_dict(),
                         torch.load(self.ckpt_last_path, map_location="cpu"))

    def test_load_checkpoint_path_state_dict(self):
        cpkt_path = os.path.join(self.test_checkpoints_path, "test_model_weights_state_dict.p")
        torch.save(torch.load(self.ckpt_1_path, map_location="cpu"), cpkt_path)  # change the ckpt path
        self.test_experiment.load_checkpoint(cpkt_path)

        self.assertEqual(self.test_experiment.model.network.state_dict(), torch.load(cpkt_path, map_location="cpu"))
