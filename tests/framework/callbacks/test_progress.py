from typing import List
from unittest import TestCase
from unittest.mock import patch, MagicMock, call

from poutyne import ProgressionCallback


class ProgressTest(TestCase):

    def setUp(self) -> None:
        self.epochs = 2
        self.num_steps = 10
        self.num_valid_steps = 10
        self.step_times_rate = 1.0
        self.any_log = {"time": self.step_times_rate, "loss": 1.0}
        self.total_time = 5
        self.final_log = {"time": self.total_time, "loss": 1.0}
        self.an_empty_log = {}

        self.metrics_str = ''

        self.coloring = True

    def test_givenATrainWithStepsValidWithoutStepsLoop_whenAlternatingLoop_thenProgressBarIsProperlySet(self):
        params = {'epochs': 2, 'steps': self.num_steps, 'valid_steps': None}

        with patch("poutyne.framework.progress.ColorProgress") as color_progress_patch:
            self.progression_callback = ProgressionCallback()
            self.progression_callback.set_params(params)
            self.progression_callback.set_model(MagicMock())

            self.progression_callback.on_train_begin(self.an_empty_log)
            color_progress_calls = [call(coloring=self.coloring), call().set_progress_bar(self.num_steps)]

            # first train loop
            epoch = 1
            color_progress_calls.extend(self._a_training_loop(epoch, self.num_steps, with_progress_bar=True))

            # first valid loop
            color_progress_calls.extend(self._a_valid_loop(epoch, self.num_steps, with_progress_bar=False))

            # second train loop
            epoch = 2
            color_progress_calls.extend(self._a_training_loop(epoch, self.num_steps, with_progress_bar=True))

            # second valid loop
            color_progress_calls.extend(self._a_valid_loop(epoch, self.num_steps, with_progress_bar=False))

            color_progress_patch.assert_has_calls(color_progress_calls)

    def test_givenATrainWithoutStepsValidWitStepsLoop_whenAlternatingLoop_thenProgressBarIsProperlySet(self):
        params = {'epochs': 2, 'steps': None, 'valid_steps': self.num_valid_steps}

        with patch("poutyne.framework.progress.ColorProgress") as color_progress_patch:
            self.progression_callback = ProgressionCallback()
            self.progression_callback.set_params(params)
            self.progression_callback.set_model(MagicMock())

            self.progression_callback.on_train_begin(self.an_empty_log)
            color_progress_calls = [call(coloring=self.coloring), call().close_progress_bar()]

            # first train loop
            epoch = 1
            color_progress_calls.extend(self._a_training_loop(epoch, self.num_steps, with_progress_bar=False))

            # first valid loop
            color_progress_calls.extend(self._a_valid_loop(epoch, self.num_steps, with_progress_bar=True))

            # second train loop
            epoch = 2
            color_progress_calls.extend(self._a_training_loop(epoch, self.num_steps, with_progress_bar=False))

            # second valid loop
            color_progress_calls.extend(self._a_valid_loop(epoch, self.num_steps, with_progress_bar=True))

            color_progress_patch.assert_has_calls(color_progress_calls)

    def test_givenATrainWithoutStepsValidWithoutStepsLoop_whenAlternatingLoop_thenProgressBarIsProperlySet(self):
        params = {'epochs': 2, 'steps': None, 'valid_steps': None}

        with patch("poutyne.framework.progress.ColorProgress") as color_progress_patch:
            self.progression_callback = ProgressionCallback()
            self.progression_callback.set_params(params)
            self.progression_callback.set_model(MagicMock())

            self.progression_callback.on_train_begin(self.an_empty_log)
            color_progress_calls = [call(coloring=self.coloring), call().close_progress_bar()]

            # first train loop
            epoch = 1
            color_progress_calls.extend(self._a_training_loop(epoch, self.num_steps, with_progress_bar=False))

            # first valid loop
            color_progress_calls.extend(self._a_valid_loop(epoch, self.num_steps, with_progress_bar=False))

            # second train loop
            epoch = 2
            color_progress_calls.extend(self._a_training_loop(epoch, self.num_steps, with_progress_bar=False))

            # second valid loop
            color_progress_calls.extend(self._a_valid_loop(epoch, self.num_steps, with_progress_bar=False))

            color_progress_patch.assert_has_calls(color_progress_calls)

    def _a_training_loop(self, epoch_number, num_steps: int = 5, with_progress_bar: bool = True) -> List:
        color_progress_calls = []

        self.progression_callback.on_epoch_begin(epoch_number, self.an_empty_log)
        if with_progress_bar:
            color_progress_calls.append(call().set_progress_bar(self.num_steps))
        else:
            color_progress_calls.append(call().close_progress_bar())
        color_progress_calls.append(call().on_epoch_begin(epoch_number=epoch_number, epochs=self.epochs))

        step_times_weighted_sum = 0.0
        for batch_number in range(0, num_steps):
            batch_number += 1  # 1 base counting
            self.progression_callback.on_train_batch_end(batch_number, self.any_log)
            if with_progress_bar:
                remaining_time = self.step_times_rate * (self.num_steps - batch_number)

                color_progress_calls.append(call().on_train_batch_end(remaining_time=remaining_time,
                                                                      batch_number=batch_number,
                                                                      metrics_str=self.metrics_str,
                                                                      steps=self.num_steps))
            else:
                step_times_weighted_sum += batch_number * self.step_times_rate
                normalizing_factor = (batch_number * (batch_number + 1)) / 2
                step_times_rate = step_times_weighted_sum / normalizing_factor

                color_progress_calls.append(call().on_train_batch_end(remaining_time=step_times_rate,
                                                                      batch_number=batch_number,
                                                                      metrics_str=self.metrics_str))
        return color_progress_calls

    def _a_valid_loop(self, epoch_number, num_steps: int = 5, with_progress_bar: bool = True) -> List:
        color_progress_calls = []

        self.progression_callback.on_valid_begin(self.an_empty_log)
        if with_progress_bar:
            color_progress_calls.append(call().set_progress_bar(self.num_steps))
        else:
            color_progress_calls.append(call().close_progress_bar())
        color_progress_calls.append(call().on_valid_begin())

        step_times_weighted_sum = 0.0
        for valid_batch_number in range(0, num_steps):
            valid_batch_number += 1  # 1 base counting
            self.progression_callback.on_valid_batch_end(valid_batch_number, self.any_log)

            if with_progress_bar:
                remaining_time = self.step_times_rate * (self.num_steps - valid_batch_number)

                color_progress_calls.append(call().on_valid_batch_end(remaining_time=remaining_time,
                                                                      batch_number=valid_batch_number,
                                                                      metrics_str=self.metrics_str,
                                                                      steps=self.num_valid_steps))
            else:
                step_times_weighted_sum += valid_batch_number * self.step_times_rate
                normalizing_factor = (valid_batch_number * (valid_batch_number + 1)) / 2
                step_times_rate = step_times_weighted_sum / normalizing_factor

                color_progress_calls.append(call().on_valid_batch_end(remaining_time=step_times_rate,
                                                                      batch_number=valid_batch_number,
                                                                      metrics_str=self.metrics_str))

        self.progression_callback.on_valid_end(self.final_log)
        self.progression_callback.on_epoch_end(epoch_number, self.final_log)

        color_progress_calls.extend([
            call().on_valid_end(total_time=self.total_time, steps=self.num_steps, metrics_str=self.metrics_str),
            call().on_epoch_end(total_time=self.total_time, train_last_steps=self.num_steps,
                                valid_last_steps=self.num_valid_steps, metrics_str=self.metrics_str)
        ])

        return color_progress_calls
