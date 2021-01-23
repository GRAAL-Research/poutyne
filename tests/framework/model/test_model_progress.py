from unittest import skipIf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from poutyne import Model, TensorDataset
from tests.framework.tools import SomeConstantEpochMetric, some_batch_metric_1, \
    some_batch_metric_2, repeat_batch_metric, some_metric_1_value, some_metric_2_value, repeat_batch_metric_value, \
    some_constant_epoch_metric_value, some_data_tensor_generator
from .base import ModelFittingTestCase
from .test_model import SomeDataGeneratorUsingStopIteration

try:
    import colorama as color
except ImportError:
    color = None


class ModelFittingTestCaseProgress(ModelFittingTestCase):
    # pylint: disable=too-many-public-methods

    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.batch_metrics = [
            some_batch_metric_1, ('custom_name', some_batch_metric_2), repeat_batch_metric, repeat_batch_metric
        ]
        self.batch_metrics_names = [
            'some_batch_metric_1', 'custom_name', 'repeat_batch_metric1', 'repeat_batch_metric2'
        ]
        self.batch_metrics_values = [
            some_metric_1_value, some_metric_2_value, repeat_batch_metric_value, repeat_batch_metric_value
        ]
        self.epoch_metrics = [SomeConstantEpochMetric()]
        self.epoch_metrics_names = ['some_constant_epoch_metric']
        self.epoch_metrics_values = [some_constant_epoch_metric_value]

        self.model = Model(self.pytorch_network,
                           self.optimizer,
                           self.loss_function,
                           batch_metrics=self.batch_metrics,
                           epoch_metrics=self.epoch_metrics)

    def assertStdoutContains(self, values):
        for value in values:
            self.assertIn(value, self.test_out.getvalue().strip())

    def assertStdoutNotContains(self, values):
        for value in values:
            self.assertNotIn(value, self.test_out.getvalue().strip())

    @skipIf(color is None, "Unable to import colorama")
    def test_fitting_with_default_coloring(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback])

        self.assertStdoutContains(["[32m", "[35m", "[36m", "[94m"])

    def test_fitting_with_progress_bar_show_epoch(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback])

        self.assertStdoutContains(["Epoch", "1/5", "2/5"])

    def test_fitting_with_progress_bar_show_steps(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback])

        self.assertStdoutContains(["steps", f"{ModelFittingTestCase.steps_per_epoch}"])

    def test_fitting_with_progress_bar_show_train_val_final_steps(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback])

        self.assertStdoutContains(["Val steps", "Train steps"])

    def test_fitting_with_no_progress_bar__dont_show_epoch(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback],
                                     verbose=False)

        self.assertStdoutNotContains(["Epoch", "1/5", "2/5"])

    @skipIf(color is None, "Unable to import colorama")
    def test_fitting_with_user_coloring(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        coloring = {
            "text_color": 'BLACK',
            "ratio_color": "BLACK",
            "metric_value_color": "BLACK",
            "time_color": "BLACK",
            "progress_bar_color": "BLACK"
        }
        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback],
                                     progress_options=dict(coloring=coloring))

        self.assertStdoutContains(["[30m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_fitting_with_user_partial_coloring(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback],
                                     progress_options=dict(coloring={
                                         "text_color": 'BLACK',
                                         "ratio_color": "BLACK"
                                     }))

        self.assertStdoutContains(["[30m", "[32m", "[35m", "[94m"])

    def test_fitting_with_user_coloring_invalid(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        with self.assertRaises(KeyError):
            _ = self.model.fit_generator(train_generator,
                                         valid_generator,
                                         epochs=ModelFittingTestCase.epochs,
                                         steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                         validation_steps=ModelFittingTestCase.steps_per_epoch,
                                         callbacks=[self.mock_callback],
                                         progress_options=dict(coloring={"invalid_name": 'A COLOR'}))

    def test_fitting_with_no_coloring(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback],
                                     progress_options=dict(coloring=False))

        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_fitting_with_progress_bar_default_color(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback],
                                     progress_options=dict(coloring=True, progress_bar=True))

        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "[94m", "\u2588"])

    @skipIf(color is None, "Unable to import colorama")
    def test_fitting_with_progress_bar_user_color(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        coloring = {
            "text_color": 'BLACK',
            "ratio_color": "BLACK",
            "metric_value_color": "BLACK",
            "time_color": "BLACK",
            "progress_bar_color": "BLACK"
        }
        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback],
                                     progress_options=dict(coloring=coloring, progress_bar=True))

        self.assertStdoutContains(["%", "[30m", "\u2588"])

    def test_fitting_with_progress_bar_no_color(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback],
                                     progress_options=dict(coloring=False, progress_bar=True))

        self.assertStdoutContains(["%", "\u2588"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    def test_fitting_with_no_progress_bar(self):
        train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                                     validation_steps=ModelFittingTestCase.steps_per_epoch,
                                     callbacks=[self.mock_callback],
                                     progress_options=dict(coloring=False, progress_bar=False))

        self.assertStdoutNotContains(["%", "\u2588"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    def test_progress_bar_with_step_is_none(self):
        train_generator = SomeDataGeneratorUsingStopIteration(ModelFittingTestCase.batch_size, 10)
        valid_generator = SomeDataGeneratorUsingStopIteration(ModelFittingTestCase.batch_size, 10)

        self._capture_output()

        _ = self.model.fit_generator(train_generator,
                                     valid_generator,
                                     epochs=ModelFittingTestCase.epochs,
                                     progress_options=dict(coloring=False, progress_bar=True))

        self.assertStdoutContains(["s/step"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m", "\u2588", "%"])

    def test_evaluate_without_progress_output(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self._capture_output()

        _, _ = self.model.evaluate(x, y, batch_size=ModelFittingTestCase.batch_size, verbose=False)

        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_default_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self._capture_output()

        _, _ = self.model.evaluate(x, y, batch_size=ModelFittingTestCase.batch_size)

        self.assertStdoutContains(["[32m", "[35m", "[36m", "[94m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_user_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self._capture_output()

        coloring = {
            "text_color": 'BLACK',
            "ratio_color": "BLACK",
            "metric_value_color": "BLACK",
            "time_color": "BLACK",
            "progress_bar_color": "BLACK"
        }

        _, _ = self.model.evaluate(x,
                                   y,
                                   batch_size=ModelFittingTestCase.batch_size,
                                   progress_options=dict(coloring=coloring))

        self.assertStdoutContains(["[30m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_user_partial_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self._capture_output()

        _, _ = self.model.evaluate(x,
                                   y,
                                   batch_size=ModelFittingTestCase.batch_size,
                                   progress_options=dict(coloring={
                                       "text_color": 'BLACK',
                                       "ratio_color": "BLACK"
                                   }))
        self.assertStdoutContains(["[30m", "[32m", "[35m", "[94m"])

    def test_evaluate_with_user_coloring_invalid(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        with self.assertRaises(KeyError):
            _, _ = self.model.evaluate(x,
                                       y,
                                       batch_size=ModelFittingTestCase.batch_size,
                                       progress_options=dict(coloring={"invalid_name": 'A COLOR'}))

    def test_evaluate_with_no_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self._capture_output()

        _, _ = self.model.evaluate(x,
                                   y,
                                   batch_size=ModelFittingTestCase.batch_size,
                                   progress_options=dict(coloring=False))

        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_progress_bar_default_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self._capture_output()

        _, _ = self.model.evaluate(x,
                                   y,
                                   batch_size=ModelFittingTestCase.batch_size,
                                   progress_options=dict(coloring=True, progress_bar=True))

        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "[94m", "\u2588"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_progress_bar_user_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self._capture_output()

        coloring = {
            "text_color": 'BLACK',
            "ratio_color": "BLACK",
            "metric_value_color": "BLACK",
            "time_color": "BLACK",
            "progress_bar_color": "BLACK"
        }

        _, _ = self.model.evaluate(x,
                                   y,
                                   batch_size=ModelFittingTestCase.batch_size,
                                   progress_options=dict(coloring=coloring, progress_bar=True))

        self.assertStdoutContains(["%", "[30m", "\u2588"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_progress_bar_user_no_color(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self._capture_output()

        _, _ = self.model.evaluate(x,
                                   y,
                                   batch_size=ModelFittingTestCase.batch_size,
                                   progress_options=dict(coloring=False, progress_bar=True))

        self.assertStdoutContains(["%", "\u2588"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    def test_evaluate_with_no_progress_bar(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self._capture_output()

        _, _ = self.model.evaluate(x,
                                   y,
                                   batch_size=ModelFittingTestCase.batch_size,
                                   progress_options=dict(coloring=False, progress_bar=False))

        self.assertStdoutNotContains(["%", "\u2588"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    def test_evaluate_data_loader_with_progress_bar_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        dataset = TensorDataset(x, y)
        generator = DataLoader(dataset, ModelFittingTestCase.batch_size)

        self._capture_output()

        _, _ = self.model.evaluate_generator(generator, verbose=True)

        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "[94m", "\u2588"])

    def test_evaluate_generator_with_progress_bar_coloring(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _, _ = self.model.evaluate_generator(generator, steps=num_steps, verbose=True)

        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "[94m", "\u2588"])

    def test_evaluate_generator_with_callback_and_progress_bar_coloring(self):
        num_steps = 10
        generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        self._capture_output()

        _, _ = self.model.evaluate_generator(generator, steps=num_steps, callbacks=[self.mock_callback], verbose=True)

        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "[94m", "\u2588"])
