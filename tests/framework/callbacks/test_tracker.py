from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf
from unittest.mock import MagicMock, call, ANY

import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from poutyne.framework import Model, Callback, TensorBoardGradientTracker


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


@skipIf(SummaryWriter is None, "Unable to import SummaryWriter from torch")
class TensorBoardGradientTrackerTest(TestCase):
    batch_size = 20
    lr = 1e-3
    num_epochs = 10

    def setUp(self):
        torch.manual_seed(42)
        self.loss_function = nn.MSELoss()
        self.temp_dir_obj = TemporaryDirectory()
        # pylint: disable=not-callable
        self.writer = SummaryWriter(self.temp_dir_obj.name)
        self.writer.add_scalars = MagicMock()

    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_tracking_one_layer_model(self):
        self.num_layer = 1
        self.pytorch_network = nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=self.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

        keep_bias = False
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        tracker = TensorBoardGradientTracker(self.writer, keep_bias=keep_bias)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[tracker])
        self._test_tracking(keep_bias)

    def test_tracking_one_layer_model_with_bias(self):
        self.num_layer = 1
        self.pytorch_network = nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=self.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

        keep_bias = True
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        tracker = TensorBoardGradientTracker(self.writer, keep_bias=keep_bias)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[tracker])
        self._test_tracking(keep_bias)

    def test_tracking_two_layers_model(self):
        self.num_layer = 2
        self.pytorch_network = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=self.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

        keep_bias = False
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        tracker = TensorBoardGradientTracker(self.writer, keep_bias=keep_bias)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[tracker])
        self._test_tracking(keep_bias)

    def test_tracking_N_layers_model_with_bias(self):
        self.num_layer = 4
        self.pytorch_network = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1))
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=self.lr)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

        keep_bias = True
        train_gen = some_data_generator(20)
        valid_gen = some_data_generator(20)
        tracker = TensorBoardGradientTracker(self.writer, keep_bias=keep_bias)
        self.model.fit_generator(train_gen, valid_gen, epochs=self.num_epochs, steps_per_epoch=5, callbacks=[tracker])
        self._test_tracking(keep_bias)

    def _test_tracking(self, keep_bias):
        expected_call = list()
        for epoch in range(1, self.num_epochs + 1):
            layer_names = [""]
            if self.num_layer > 1:
                layer_names = []
                for layer_idx in range(self.num_layer):
                    layer_names.append("{}.".format(layer_idx))
            for layer_name in layer_names:
                expected_call.append(call('gradient_distributions/{}weight'.format(layer_name), {'mean': ANY}, epoch))
                expected_call.append(
                    call('gradient_distributions/{}weight'.format(layer_name), {'std_dev_up': ANY}, epoch))
                expected_call.append(
                    call('gradient_distributions/{}weight'.format(layer_name), {'std_dev_down': ANY}, epoch))
                expected_call.append(call('other_gradient_stats/{}weight'.format(layer_name), {'min': ANY}, epoch))
                expected_call.append(call('other_gradient_stats/{}weight'.format(layer_name), {'max': ANY}, epoch))

                if keep_bias:
                    expected_call.append(call('gradient_distributions/{}bias'.format(layer_name), {'mean': ANY}, epoch))
                    expected_call.append(
                        call('gradient_distributions/{}bias'.format(layer_name), {'std_dev_up': ANY}, epoch))
                    expected_call.append(
                        call('gradient_distributions/{}bias'.format(layer_name), {'std_dev_down': ANY}, epoch))
                    expected_call.append(call('other_gradient_stats/{}bias'.format(layer_name), {'min': ANY}, epoch))
                    expected_call.append(call('other_gradient_stats/{}bias'.format(layer_name), {'max': ANY}, epoch))

        method_calls = self.writer.add_scalars.mock_calls
        self.assertEqual(len(method_calls), len(expected_call))
        self.assertEqual(method_calls, expected_call)
