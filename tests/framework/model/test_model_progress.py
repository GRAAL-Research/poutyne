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

from unittest import skipIf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from poutyne import Model, TensorDataset
from tests.framework.tools import (
    SomeConstantMetric,
    some_batch_metric_1,
    some_batch_metric_2,
    repeat_batch_metric,
    some_metric_1_value,
    some_metric_2_value,
    repeat_batch_metric_value,
    some_constant_metric_value,
    some_data_tensor_generator,
    SomeDataGeneratorUsingStopIteration,
)
from .base import ModelFittingTestCase

try:
    import colorama as color
except ImportError:
    color = None


class ModelFittingTestCaseProgress(ModelFittingTestCase):
    # pylint: disable=too-many-public-methods
    num_steps = 5
    TIME_REGEX = r"((([0-9]+d)?[0-9]{1,2}h)?[0-9]{1,2}m)?[0-9]{1,2}\.[0-9]{2}s"

    def setUp(self):
        super().setUp()
        self.train_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        self.valid_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        self.test_generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)
        torch.manual_seed(42)
        self.pytorch_network = nn.Linear(1, 1)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.pytorch_network.parameters(), lr=1e-3)
        self.batch_metrics = [
            some_batch_metric_1,
            ('custom_name', some_batch_metric_2),
            repeat_batch_metric,
            repeat_batch_metric,
        ]
        self.batch_metrics_names = [
            'some_batch_metric_1',
            'custom_name',
            'repeat_batch_metric1',
            'repeat_batch_metric2',
        ]
        self.batch_metrics_values = [
            some_metric_1_value,
            some_metric_2_value,
            repeat_batch_metric_value,
            repeat_batch_metric_value,
        ]
        self.epoch_metrics = [SomeConstantMetric()]
        self.epoch_metrics_names = ['some_constant_metric']
        self.epoch_metrics_values = [some_constant_metric_value]

        self.model = Model(
            self.pytorch_network,
            self.optimizer,
            self.loss_function,
            batch_metrics=self.batch_metrics,
            epoch_metrics=self.epoch_metrics,
        )

        self._capture_output()

    def assertStdoutContains(self, values):
        for value in values:
            self.assertIn(value, self.test_out.getvalue().strip())

    def assertStdoutNotContains(self, values):
        for value in values:
            self.assertNotIn(value, self.test_out.getvalue().strip())

    @skipIf(color is None, "Unable to import colorama")
    def test_fitting_with_default_coloring(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
        )

        self.assertStdoutContains(["[32m", "[35m", "[36m", "[94m"])

    def test_fitting_with_progress_bar_show_epoch(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
        )

        self.assertStdoutContains(["Epoch", "1/5", "2/5"])

    def test_fitting_with_progress_bar_show_steps(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
        )

        self.assertStdoutContains(["steps", f"{ModelFittingTestCase.steps_per_epoch}"])

    def test_fitting_with_progress_bar_show_train_val_final_steps(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
        )

        self.assertStdoutContains(["Val steps", "Train steps"])

    def test_fitting_with_no_progress_bar_dont_show_epoch(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
            verbose=False,
        )

        self.assertStdoutNotContains(["Epoch", "1/5", "2/5"])

    @skipIf(color is None, "Unable to import colorama")
    def test_fitting_with_user_coloring(self):
        coloring = {
            "text_color": 'BLACK',
            "ratio_color": "BLACK",
            "metric_value_color": "BLACK",
            "time_color": "BLACK",
            "progress_bar_color": "BLACK",
        }
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
            progress_options=dict(coloring=coloring),
        )

        self.assertStdoutContains(["[30m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_fitting_with_user_partial_coloring(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
            progress_options=dict(coloring={"text_color": 'BLACK', "ratio_color": "BLACK"}),
        )

        self.assertStdoutContains(["[30m", "[32m", "[35m", "[94m"])

    def test_fitting_with_user_coloring_invalid(self):
        with self.assertRaises(KeyError):
            _ = self.model.fit_generator(
                self.train_generator,
                self.valid_generator,
                epochs=ModelFittingTestCase.epochs,
                steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
                validation_steps=ModelFittingTestCase.steps_per_epoch,
                callbacks=[self.mock_callback],
                progress_options=dict(coloring={"invalid_name": 'A COLOR'}),
            )

    def test_fitting_with_no_coloring(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
            progress_options=dict(coloring=False),
        )

        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_fitting_with_progress_bar_default_color(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
            progress_options=dict(coloring=True, progress_bar=True),
        )

        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "[94m", "\u2588"])

    @skipIf(color is None, "Unable to import colorama")
    def test_fitting_with_progress_bar_user_color(self):
        coloring = {
            "text_color": 'BLACK',
            "ratio_color": "BLACK",
            "metric_value_color": "BLACK",
            "time_color": "BLACK",
            "progress_bar_color": "BLACK",
        }
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
            progress_options=dict(coloring=coloring, progress_bar=True),
        )

        self.assertStdoutContains(["%", "[30m", "\u2588"])

    def test_fitting_with_progress_bar_no_color(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
            progress_options=dict(coloring=False, progress_bar=True),
        )

        self.assertStdoutContains(["%", "\u2588"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    def test_fitting_with_no_progress_bar(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=ModelFittingTestCase.epochs,
            steps_per_epoch=ModelFittingTestCase.steps_per_epoch,
            validation_steps=ModelFittingTestCase.steps_per_epoch,
            callbacks=[self.mock_callback],
            progress_options=dict(coloring=False, progress_bar=False),
        )

        self.assertStdoutNotContains(["%", "\u2588"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    def test_progress_bar_with_step_is_none(self):
        train_generator = SomeDataGeneratorUsingStopIteration(ModelFittingTestCase.batch_size, 10)
        valid_generator = SomeDataGeneratorUsingStopIteration(ModelFittingTestCase.batch_size, 10)
        _ = self.model.fit_generator(
            train_generator,
            valid_generator,
            epochs=ModelFittingTestCase.epochs,
            progress_options=dict(coloring=False, progress_bar=True),
        )

        self.assertStdoutContains(["s/step"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m", "\u2588", "%"])

    def test_evaluate_without_progress_output(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        _, _ = self.model.evaluate(x, y, batch_size=ModelFittingTestCase.batch_size, verbose=False)

        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_default_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        _, _ = self.model.evaluate(x, y, batch_size=ModelFittingTestCase.batch_size)

        self.assertStdoutContains(["[32m", "[35m", "[36m", "[94m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_user_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        coloring = {
            "text_color": 'BLACK',
            "ratio_color": "BLACK",
            "metric_value_color": "BLACK",
            "time_color": "BLACK",
            "progress_bar_color": "BLACK",
        }

        _, _ = self.model.evaluate(
            x, y, batch_size=ModelFittingTestCase.batch_size, progress_options=dict(coloring=coloring)
        )

        self.assertStdoutContains(["[30m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_user_partial_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        _, _ = self.model.evaluate(
            x,
            y,
            batch_size=ModelFittingTestCase.batch_size,
            progress_options=dict(coloring={"text_color": 'BLACK', "ratio_color": "BLACK"}),
        )
        self.assertStdoutContains(["[30m", "[32m", "[35m", "[94m"])

    def test_evaluate_with_user_coloring_invalid(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        with self.assertRaises(KeyError):
            _, _ = self.model.evaluate(
                x,
                y,
                batch_size=ModelFittingTestCase.batch_size,
                progress_options=dict(coloring={"invalid_name": 'A COLOR'}),
            )

    def test_evaluate_with_no_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        _, _ = self.model.evaluate(
            x, y, batch_size=ModelFittingTestCase.batch_size, progress_options=dict(coloring=False)
        )

        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_progress_bar_default_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        _, _ = self.model.evaluate(
            x, y, batch_size=ModelFittingTestCase.batch_size, progress_options=dict(coloring=True, progress_bar=True)
        )

        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "[94m", "\u2588"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_progress_bar_user_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        coloring = {
            "text_color": 'BLACK',
            "ratio_color": "BLACK",
            "metric_value_color": "BLACK",
            "time_color": "BLACK",
            "progress_bar_color": "BLACK",
        }

        _, _ = self.model.evaluate(
            x,
            y,
            batch_size=ModelFittingTestCase.batch_size,
            progress_options=dict(coloring=coloring, progress_bar=True),
        )

        self.assertStdoutContains(["%", "[30m", "\u2588"])

    @skipIf(color is None, "Unable to import colorama")
    def test_evaluate_with_progress_bar_user_no_color(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        _, _ = self.model.evaluate(
            x, y, batch_size=ModelFittingTestCase.batch_size, progress_options=dict(coloring=False, progress_bar=True)
        )

        self.assertStdoutContains(["%", "\u2588"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    def test_evaluate_with_no_progress_bar(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        _, _ = self.model.evaluate(
            x, y, batch_size=ModelFittingTestCase.batch_size, progress_options=dict(coloring=False, progress_bar=False)
        )

        self.assertStdoutNotContains(["%", "\u2588"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m", "[94m"])

    def test_evaluate_data_loader_with_progress_bar_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        y = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        dataset = TensorDataset(x, y)
        generator = DataLoader(dataset, ModelFittingTestCase.batch_size)

        _, _ = self.model.evaluate_generator(generator, verbose=True)

        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "[94m", "\u2588"])

    def test_evaluate_generator_with_progress_bar_coloring(self):
        generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        _, _ = self.model.evaluate_generator(generator, steps=ModelFittingTestCaseProgress.num_steps, verbose=True)

        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "[94m", "\u2588"])

    def test_evaluate_generator_with_callback_and_progress_bar_coloring(self):
        generator = some_data_tensor_generator(ModelFittingTestCase.batch_size)

        _, _ = self.model.evaluate_generator(
            generator, steps=ModelFittingTestCaseProgress.num_steps, callbacks=[self.mock_callback], verbose=True
        )

        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "[94m", "\u2588"])

    def test_fitting_complete_display_test_with_progress_bar_coloring(self):
        # we use the same color for all components for simplicity
        coloring = {
            "text_color": 'WHITE',
            "ratio_color": "WHITE",
            "metric_value_color": "WHITE",
            "time_color": "WHITE",
            "progress_bar_color": "WHITE",
        }
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=1,
            steps_per_epoch=ModelFittingTestCaseProgress.num_steps,
            validation_steps=ModelFittingTestCaseProgress.num_steps,
            callbacks=[self.mock_callback],
            progress_options=dict(coloring=coloring, progress_bar=False),
        )

        # We split per step update
        steps_update = self.test_out.getvalue().strip().split("\r")

        # we don't validate the templating of metrics since tested before
        template_format = r".*Epoch:.*{}\/1.*\[37mStep:.*{}\/5.*{:6.2f}\%.*|{}|.*ETA:"
        epoch = 1
        # the 5 train steps
        for step, step_update in enumerate(steps_update[: ModelFittingTestCaseProgress.num_steps]):
            step += 1
            progress_Bar = "\u2588" * step * 2 + " " * (20 - step * 2)
            regex_filled = template_format.format(
                epoch, step, step / ModelFittingTestCaseProgress.num_steps * 100, progress_Bar
            )
            self.assertRegex(step_update, regex_filled)

        # The 5 val steps
        for step, step_update in enumerate(steps_update[ModelFittingTestCaseProgress.num_steps : -1]):
            step += 1
            progress_Bar = "\u2588" * step * 2 + " " * (20 - step * 2)
            regex_filled = template_format.format(
                epoch, step, step / ModelFittingTestCaseProgress.num_steps * 100, progress_Bar
            )
            self.assertRegex(step_update, regex_filled)

        # last print update templating different
        last_print_regex = r".*\[37mTrain steps:.*5.*Val steps:.*5.*" + ModelFittingTestCaseProgress.TIME_REGEX
        self.assertRegex(steps_update[-1], last_print_regex)

    def test_fitting_complete_display_test_with_progress_bar_no_coloring(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=1,
            steps_per_epoch=ModelFittingTestCaseProgress.num_steps,
            validation_steps=ModelFittingTestCaseProgress.num_steps,
            callbacks=[self.mock_callback],
            progress_options=dict(coloring=False, progress_bar=True),
        )

        # We split per step update
        steps_update = self.test_out.getvalue().strip().split("\r")

        # we don't validate the templating of metrics since tested before
        template_format = r".*Epoch:.*{}\/1.*Step:.*{}\/5.*{:6.2f}\%.*|{}|.*ETA:"
        epoch = 1
        # the 5 train steps
        for step, step_update in enumerate(steps_update[: ModelFittingTestCaseProgress.num_steps]):
            step += 1
            progress_Bar = "\u2588" * step * 2 + " " * (20 - step * 2)
            regex_filled = template_format.format(
                epoch, step, step / ModelFittingTestCaseProgress.num_steps * 100, progress_Bar
            )
            self.assertRegex(step_update, regex_filled)

        # The 5 val steps
        for step, step_update in enumerate(steps_update[ModelFittingTestCaseProgress.num_steps : -1]):
            step += 1
            progress_Bar = "\u2588" * step * 2 + " " * (20 - step * 2)
            regex_filled = template_format.format(
                epoch, step, step / ModelFittingTestCaseProgress.num_steps * 100, progress_Bar
            )
            self.assertRegex(step_update, regex_filled)

        # last print update templating different
        last_print_regex = r".*Train steps:.*5.*Val steps:.*5.*" + ModelFittingTestCaseProgress.TIME_REGEX
        self.assertRegex(steps_update[-1], last_print_regex)

    def test_fitting_complete_display_test_with_no_progress_bar_no_coloring(self):
        _ = self.model.fit_generator(
            self.train_generator,
            self.valid_generator,
            epochs=1,
            steps_per_epoch=ModelFittingTestCaseProgress.num_steps,
            validation_steps=ModelFittingTestCaseProgress.num_steps,
            callbacks=[self.mock_callback],
            progress_options=dict(coloring=False, progress_bar=False),
        )

        # We split per step update
        steps_update = self.test_out.getvalue().strip().split("\r")

        # we don't validate the templating of metrics since tested before
        template_format = r".*Epoch:.*{}\/1.*Step:.*{}\/5.*ETA:"
        epoch = 1
        # the 5 train steps
        for step, step_update in enumerate(steps_update[: ModelFittingTestCaseProgress.num_steps]):
            step += 1
            regex_filled = template_format.format(epoch, step, step / ModelFittingTestCaseProgress.num_steps * 100)
            self.assertRegex(step_update, regex_filled)

        # The 5 val steps
        for step, step_update in enumerate(steps_update[ModelFittingTestCaseProgress.num_steps : -1]):
            step += 1
            regex_filled = template_format.format(epoch, step, step / ModelFittingTestCaseProgress.num_steps * 100)
            self.assertRegex(step_update, regex_filled)

        # last print update templating different
        last_print_regex = r".*Train steps:.*5.*Val steps:.*5.*" + ModelFittingTestCaseProgress.TIME_REGEX
        self.assertRegex(steps_update[-1], last_print_regex)

    def test_evaluate_complete_display_test_with_progress_bar_coloring(self):
        # we use the same color for all components for simplicity
        coloring = {
            "text_color": 'WHITE',
            "ratio_color": "WHITE",
            "metric_value_color": "WHITE",
            "time_color": "WHITE",
            "progress_bar_color": "WHITE",
        }

        _, _ = self.model.evaluate_generator(
            self.test_generator,
            steps=ModelFittingTestCaseProgress.num_steps,
            callbacks=[self.mock_callback],
            verbose=True,
            progress_options=dict(coloring=coloring, progress_bar=True),
        )

        # We split per step update
        steps_update = self.test_out.getvalue().strip().split("\r")

        # we don't validate the templating of metrics since tested before
        template_format = r".*\[37mStep:.*{}\/5.*{:6.2f}\%.*|{}|.*ETA:"
        for step, step_update in enumerate(steps_update[:-1]):
            step += 1
            progress_Bar = "\u2588" * step * 2 + " " * (20 - step * 2)
            regex_filled = template_format.format(
                step, step / ModelFittingTestCaseProgress.num_steps * 100, progress_Bar
            )
            self.assertRegex(step_update, regex_filled)

        # last print update templating different
        last_print_regex = r".*\[37mTest steps:.*5.*" + ModelFittingTestCaseProgress.TIME_REGEX
        self.assertRegex(steps_update[-1], last_print_regex)

    def test_evaluate_complete_display_test_with_progress_bar_no_coloring(self):
        _, _ = self.model.evaluate_generator(
            self.test_generator,
            steps=ModelFittingTestCaseProgress.num_steps,
            callbacks=[self.mock_callback],
            verbose=True,
            progress_options=dict(coloring=False, progress_bar=True),
        )

        # We split per step update
        steps_update = self.test_out.getvalue().strip().split("\r")

        # we don't validate the templating of metrics since tested before
        template_format = r".*Step:.*{}\/5.*{:6.2f}\%.*|{}|.*ETA:"
        for step, step_update in enumerate(steps_update[:-1]):
            step += 1
            progress_Bar = "\u2588" * step * 2 + " " * (20 - step * 2)
            regex_filled = template_format.format(
                step, step / ModelFittingTestCaseProgress.num_steps * 100, progress_Bar
            )
            self.assertRegex(step_update, regex_filled)

        # last print update templating different
        last_print_regex = r".*Test steps:.*5.*" + ModelFittingTestCaseProgress.TIME_REGEX
        self.assertRegex(steps_update[-1], last_print_regex)

    def test_evaluate_complete_display_test_with_no_progress_bar_no_coloring(self):
        _, _ = self.model.evaluate_generator(
            self.test_generator,
            steps=ModelFittingTestCaseProgress.num_steps,
            callbacks=[self.mock_callback],
            verbose=True,
            progress_options=dict(coloring=False, progress_bar=False),
        )

        # We split per step update
        steps_update = self.test_out.getvalue().strip().split("\r")

        # we don't validate the templating of metrics since tested before
        template_format = r".*Step:.*{}\/5.*ETA:"
        for step, step_update in enumerate(steps_update[:-1]):
            step += 1
            regex_filled = template_format.format(step, step / ModelFittingTestCaseProgress.num_steps * 100)
            self.assertRegex(step_update, regex_filled)

        # last print update templating different
        last_print_regex = r".*Test steps:.*5.*" + ModelFittingTestCaseProgress.TIME_REGEX
        self.assertRegex(steps_update[-1], last_print_regex)

    @skipIf(color is None, "Unable to import colorama")
    def test_predict_dataset_with_default_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self.model.predict_dataset(x)

        self.assertStdoutContains(["[32m", "[35m", "[36m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_predict_dataset_with_user_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        coloring = {
            "text_color": 'BLACK',
            "ratio_color": "BLACK",
            "metric_value_color": "BLACK",
            "time_color": "BLACK",
            "progress_bar_color": "BLACK",
        }

        self.model.predict_dataset(x, progress_options=dict(coloring=coloring, progress_bar=True))

        self.assertStdoutContains(["[30m"])

    def test_predict_dataset_with_user_coloring_invalid(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        with self.assertRaises(KeyError):
            self.model.predict_dataset(
                x,
                batch_size=ModelFittingTestCase.batch_size,
                progress_options=dict(coloring={"invalid_name": 'A COLOR'}),
            )

    def test_predict_dataset_with_no_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self.model.predict_dataset(x, batch_size=ModelFittingTestCase.batch_size, progress_options=dict(coloring=False))

        self.assertStdoutNotContains(["[32m", "[35m", "[36m"])

    @skipIf(color is None, "Unable to import colorama")
    def test_predict_dataset_with_progress_bar_default_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self.model.predict_dataset(
            x, batch_size=ModelFittingTestCase.batch_size, progress_options=dict(coloring=True, progress_bar=True)
        )

        self.assertStdoutContains(["%", "[32m", "[35m", "[36m", "\u2588"])

    @skipIf(color is None, "Unable to import colorama")
    def test_predict_dataset_with_progress_bar_user_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        coloring = {
            "text_color": 'BLACK',
            "ratio_color": "BLACK",
            "metric_value_color": "BLACK",
            "time_color": "BLACK",
            "progress_bar_color": "BLACK",
        }

        self.model.predict_dataset(
            x, batch_size=ModelFittingTestCase.batch_size, progress_options=dict(coloring=coloring, progress_bar=True)
        )

        self.assertStdoutContains(["%", "[30m", "\u2588"])

    @skipIf(color is None, "Unable to import colorama")
    def test_predict_dataset_with_progress_bar_user_no_color(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self.model.predict_dataset(
            x, batch_size=ModelFittingTestCase.batch_size, progress_options=dict(coloring=False, progress_bar=True)
        )

        self.assertStdoutContains(["%", "\u2588"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m"])

    def test_predict_dataset_with_no_progress_bar(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)

        self.model.predict_dataset(
            x, batch_size=ModelFittingTestCase.batch_size, progress_options=dict(coloring=False, progress_bar=False)
        )

        self.assertStdoutNotContains(["%", "\u2588"])
        self.assertStdoutNotContains(["[32m", "[35m", "[36m"])

    def test_predict_dataset_complete_display_predict_with_progress_bar_coloring(self):
        x = torch.rand(ModelFittingTestCase.evaluate_dataset_len, 1)
        # we use the same color for all components for simplicity
        coloring = {
            "text_color": 'WHITE',
            "ratio_color": "WHITE",
            "metric_value_color": "WHITE",
            "time_color": "WHITE",
            "progress_bar_color": "WHITE",
        }

        self.model.predict_dataset(x, verbose=True, progress_options=dict(coloring=coloring, progress_bar=True))

        # We split per step update
        steps_update = self.test_out.getvalue().strip().split("\r")

        # last print update templating different
        last_print_regex = r".*\[37mPrediction steps:.*" + ModelFittingTestCaseProgress.TIME_REGEX
        self.assertRegex(steps_update[-1], last_print_regex)
