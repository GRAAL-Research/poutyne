from unittest.mock import Mock, call

import torch
import torch.nn as nn

from poutyne import Model, LambdaCallback, Callback
from tests.framework.tools import some_data_tensor_generator
from tests.framework.model.base import ModelFittingTestCase


class LambdaTest(ModelFittingTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.pytorch_network.parameters(), lr=1e-3)
        self.model = Model(self.pytorch_network, self.optimizer, self.loss_function)

    def test_integration_zero_args(self):
        lambda_callback = LambdaCallback()

        train_generator = some_data_tensor_generator(LambdaTest.batch_size)
        valid_generator = some_data_tensor_generator(LambdaTest.batch_size)
        test_generator = some_data_tensor_generator(LambdaTest.batch_size)
        self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=LambdaTest.epochs,
            steps_per_epoch=LambdaTest.steps_per_epoch,
            validation_steps=LambdaTest.steps_per_epoch,
            callbacks=[lambda_callback],
        )

        num_steps = 10
        self.model.evaluate_generator(test_generator, steps=num_steps, callbacks=[lambda_callback])

    def test_with_only_on_epoch_end_arg(self):
        on_epoch_end = Mock()
        lambda_callback = LambdaCallback(on_epoch_end=on_epoch_end)

        train_generator = some_data_tensor_generator(LambdaTest.batch_size)
        valid_generator = some_data_tensor_generator(LambdaTest.batch_size)
        test_generator = some_data_tensor_generator(LambdaTest.batch_size)
        logs = self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=LambdaTest.epochs,
            steps_per_epoch=LambdaTest.steps_per_epoch,
            validation_steps=LambdaTest.steps_per_epoch,
            callbacks=[lambda_callback],
        )

        num_steps = 10
        self.model.evaluate_generator(test_generator, steps=num_steps, callbacks=[lambda_callback])

        expected_calls = [call(epoch_number, log) for epoch_number, log in enumerate(logs, 1)]
        actual_calls = on_epoch_end.mock_calls
        self.assertEqual(len(expected_calls), len(actual_calls))
        self.assertEqual(expected_calls, actual_calls)

    def test_lambda_test_calls(self):
        lambda_callback, mock_calls = self._get_lambda_callback_with_mock_args()
        num_steps = 10
        generator = some_data_tensor_generator(LambdaTest.batch_size)
        self.model.evaluate_generator(generator, steps=num_steps, callbacks=[lambda_callback, self.mock_callback])

        expected_calls = self.mock_callback.method_calls[2:]
        actual_calls = mock_calls.method_calls
        self.assertEqual(len(expected_calls), len(actual_calls))
        self.assertEqual(expected_calls, actual_calls)

    def test_lambda_train_calls(self):
        lambda_callback, mock_calls = self._get_lambda_callback_with_mock_args()
        train_generator = some_data_tensor_generator(LambdaTest.batch_size)
        valid_generator = some_data_tensor_generator(LambdaTest.batch_size)
        self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=LambdaTest.epochs,
            steps_per_epoch=LambdaTest.steps_per_epoch,
            validation_steps=LambdaTest.steps_per_epoch,
            callbacks=[lambda_callback, self.mock_callback],
        )

        expected_calls = self.mock_callback.method_calls[2:]
        actual_calls = mock_calls.method_calls
        self.assertEqual(len(expected_calls), len(actual_calls))
        self.assertEqual(expected_calls, actual_calls)

    def _get_lambda_callback_with_mock_args(self):
        mock_callback = Mock(spec=Callback())
        lambda_callback = LambdaCallback(
            on_epoch_begin=mock_callback.on_epoch_begin,
            on_epoch_end=mock_callback.on_epoch_end,
            on_train_batch_begin=mock_callback.on_train_batch_begin,
            on_train_batch_end=mock_callback.on_train_batch_end,
            on_valid_batch_begin=mock_callback.on_valid_batch_begin,
            on_valid_batch_end=mock_callback.on_valid_batch_end,
            on_test_batch_begin=mock_callback.on_test_batch_begin,
            on_test_batch_end=mock_callback.on_test_batch_end,
            on_train_begin=mock_callback.on_train_begin,
            on_train_end=mock_callback.on_train_end,
            on_valid_begin=mock_callback.on_valid_begin,
            on_valid_end=mock_callback.on_valid_end,
            on_test_begin=mock_callback.on_test_begin,
            on_test_end=mock_callback.on_test_end,
            on_backward_end=mock_callback.on_backward_end,
        )
        return lambda_callback, mock_callback
