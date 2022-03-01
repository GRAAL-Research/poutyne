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

import math
import sys
import time
import warnings
from typing import Dict, Union

from poutyne import is_in_jupyter_notebook
from .progress_bar import ProgressBar


class EmptyStringAttrClass:
    """
    Class to emulate the Fore and Style class of colorama with a class that as an empty string for every attributes.
    """

    def __getattr__(self, attr):
        return ''


jupyter = is_in_jupyter_notebook()

try:
    from colorama import Fore, Style, init

    colorama = True

    # We don't init when Jupyter Notebook see issue https://github.com/jupyter/notebook/issues/2284
    if not jupyter:
        init()
except ImportError:
    colorama = None

    Fore = EmptyStringAttrClass()
    Style = EmptyStringAttrClass()

default_color_settings = {
    "text_color": 'MAGENTA',
    "ratio_color": "CYAN",
    "metric_value_color": "LIGHTBLUE_EX",
    "time_color": "GREEN",
    "progress_bar_color": "MAGENTA",
}


class ColorProgress:
    JUPYTER_PRINT_RATE = 0.1
    """
    Class to managed the color templating of the training progress.

    Args:
          coloring (Union[bool, Dict], optional): If bool, whether to display the progress of the training with
                default colors highlighting.
                If Dict, the field and the color to use as colorama <https://pypi.org/project/colorama/>`_ . The fields
                are text_color, ratio_color, metric_value_color and time_color.
                In both case, will be ignore if verbose is set to False.
                (Default value = True)
    Attributes:
        text_color (str): The color to use for the text.
        ratio_color (str): The color to use for the ratio.
        metric_value_color (str): The color to use for the metric value.
        time_color (str): The color to use for the time.
        progress_bar_color (str): The color to use for the progress bar.

    """

    def __init__(self, coloring: Union[bool, Dict]) -> None:
        color_settings = None

        if (isinstance(coloring, Dict) or coloring) and colorama is None:
            warnings.warn("The colorama package was not imported. Consider installing it for colorlog.", ImportWarning)

        if isinstance(coloring, Dict):
            color_settings = default_color_settings.copy()

            invalid_keys = coloring.keys() - color_settings.keys()
            if len(invalid_keys) != 0:
                raise KeyError(f"The key(s) {', '.join(invalid_keys)} are not supported color attributes.")

            color_settings.update(coloring)
        elif coloring is True and colorama is not None:
            color_settings = default_color_settings

        if color_settings is not None:
            self.coloring_enabled = True
            self.style_reset = True
            self.text_color = getattr(Fore, color_settings["text_color"])
            self.ratio_color = getattr(Fore, color_settings["ratio_color"])
            self.metric_value_color = getattr(Fore, color_settings["metric_value_color"])
            self.time_color = getattr(Fore, color_settings["time_color"])
            self.progress_bar_color = getattr(Fore, color_settings["progress_bar_color"])
        else:
            self.coloring_enabled = False
            self.style_reset = False
            self.text_color = ""
            self.ratio_color = ""
            self.metric_value_color = ""
            self.time_color = ""
            self.progress_bar_color = ""

        self.progress_bar = False
        self.steps_progress_bar = None
        self.formatted_text = "\r"
        self.bar_format = f"{self.text_color}{{percentage}} |{self.progress_bar_color}{{bar}}{self.text_color}|"
        self.prev_print_time = None
        self.prev_message_length = 0

    def on_valid_begin(self) -> None:
        self._on_phase_begin()

    def on_test_begin(self) -> None:
        self._on_phase_begin()

    def on_predict_begin(self) -> None:
        self._on_phase_begin()

    def _on_phase_begin(self) -> None:
        if self.progress_bar:
            self.steps_progress_bar.reset()

    def on_epoch_begin(self, epoch_number: int, epochs: int) -> None:
        if self.progress_bar:
            self.steps_progress_bar.reset()

        self._set_epoch_formatted_text(epoch_number, epochs)

    def on_train_batch_end(
        self,
        *,
        remaining_time: float,
        batch_number: int,
        metrics_str: str,
        steps: Union[int, None] = None,
        do_print: bool = True,
    ) -> None:
        """
        Format on train batch end for a steps the epoch ratio (so far / to do), the total time for the epoch, the steps
        done and the metrics name and values.
        """
        update = self.epoch_formatted_text
        self._on_batch_end(update, remaining_time, batch_number, metrics_str, steps, do_print)

    def on_valid_batch_end(
        self,
        *,
        remaining_time: float,
        batch_number: int,
        metrics_str: str,
        steps: Union[int, None] = None,
        do_print: bool = True,
    ) -> None:
        """
        Format on valid batch end for a steps the epoch ratio (so far / to do), the total time, the steps
        done and the metrics name and values.
        """
        update = self.epoch_formatted_text
        self._on_batch_end(update, remaining_time, batch_number, metrics_str, steps, do_print)

    def on_test_batch_end(
        self,
        *,
        remaining_time: float,
        batch_number: int,
        metrics_str: str,
        steps: Union[int, None] = None,
        do_print: bool = True,
    ) -> None:
        """
        Format on test batch end for a steps the epoch ratio (so far / to do), the total time, the steps
        done and the metrics name and values.
        """
        update = self.formatted_text
        self._on_batch_end(update, remaining_time, batch_number, metrics_str, steps, do_print)

    def on_predict_batch_end(
        self,
        *,
        remaining_time: float,
        batch_number: int,
        metrics_str: str,
        steps: Union[int, None] = None,
        do_print: bool = True,
    ) -> None:
        """
        Format on predict batch end for a steps, the total time and the steps ratio done.
        """
        update = self.formatted_text
        self._on_batch_end(update, remaining_time, batch_number, metrics_str, steps, do_print)

    def _on_batch_end(
        self,
        update: str,
        remaining_time: float,
        batch_number: int,
        metrics_str: str,
        steps: Union[int, None] = None,
        do_print: bool = True,
    ) -> None:
        # pylint: disable=too-many-arguments
        update += self._batch_update(remaining_time, batch_number, metrics_str, steps)

        if do_print:
            self._update_print(update)

    def on_epoch_end(self, total_time: float, train_last_steps: int, valid_last_steps: int, metrics_str: str) -> None:
        """
        Format on epoch end: the epoch ratio (so far / to do), the total time for the epoch, the steps done and the
        metrics name and values.
        """
        update = self.epoch_formatted_text
        steps_text = self._get_formatted_step(
            train_last_steps, train_last_steps, prefix="train ", suffix="s", ratio=False
        )
        if valid_last_steps is not None:
            valid_steps = self._get_formatted_step(
                valid_last_steps, valid_last_steps, prefix="val ", suffix="s", ratio=False
            )
            steps_text += valid_steps
        update += steps_text + self._get_formatted_total_time(total_time)
        update += self._get_formatted_metrics(metrics_str)

        if self.style_reset:
            update += Style.RESET_ALL

        self._end_print(update)

    def on_test_end(self, total_time: float, steps: int, metrics_str: str) -> None:
        """
        Format on test end: the total time for the test, the steps done and the metrics name and values.
        """
        self.on_phase_end(total_time=total_time, steps=steps, metrics_str=metrics_str, prefix="Test ")

    def on_predict_end(self, total_time: float, steps: int, metrics_str: str) -> None:
        """
        Format on predict end: the total time for the predictions and the steps done.
        """
        self.on_phase_end(total_time=total_time, steps=steps, metrics_str=metrics_str, prefix="Prediction ")

    def on_phase_end(self, total_time: float, steps: int, metrics_str: str, prefix: str = "") -> None:
        """
        Format on a end phase: the total time for the test, the steps done and the metrics name and values.
        """
        update = self.formatted_text
        test_steps = self._get_formatted_step(steps, steps, prefix=prefix, suffix="s", ratio=False)
        update += test_steps + self._get_formatted_total_time(total_time)
        if metrics_str != "":
            update += self._get_formatted_metrics(metrics_str)

        if self.style_reset:
            update += Style.RESET_ALL

        self._end_print(update)

    def set_progress_bar(self, number_steps_per_epoch: int) -> None:
        self.steps_progress_bar = ProgressBar(number_steps_per_epoch, bar_format=self.bar_format)
        self.progress_bar = True

    def close_progress_bar(self) -> None:
        self.steps_progress_bar = None
        self.progress_bar = False

    def _set_epoch_formatted_text(self, epoch_number: int, epochs: int) -> None:
        digits = int(math.log10(epochs)) + 1
        self.epoch_formatted_text = f"\r{self.text_color}Epoch: {self.ratio_color}{epoch_number:{digits}d}/{epochs:d} "

    def _format_duration(self, duration):
        days, rem = divmod(duration, 60 * 60 * 24)
        hours, rem = divmod(rem, 60 * 60)
        minutes, seconds = divmod(rem, 60)
        days = int(days)
        hours = int(hours)
        minutes = int(minutes)
        ret = ""
        if days > 0:
            ret += f"{days}d"
        if days > 0 or hours > 0:
            ret += f"{hours}h"
        if days > 0 or hours > 0 or minutes > 0:
            ret += f"{minutes}m"
        ret += f"{seconds:.2f}s"
        return ret

    def _get_formatted_total_time(self, total_time: float) -> str:
        total_time_str = self._format_duration(total_time)
        return f"{self.time_color}{total_time_str} "

    def _get_formatted_time(self, duration: float, steps: Union[int, None]) -> str:
        duration_str = self._format_duration(duration)
        if steps is None:
            formatted_time = f"{self.time_color}{duration_str}/step "
        else:
            formatted_time = f"{self.text_color}ETA: {self.time_color}{duration_str} "
        return formatted_time

    def _get_formatted_step(
        self, batch_number: int, steps: Union[int, None], prefix: str = "", suffix: str = "", ratio: bool = True
    ) -> str:
        # pylint: disable=too-many-arguments
        step_text = f"{prefix}step{suffix}".capitalize()
        ratio_text = ""
        if steps is None:
            formatted_step = f"{self.text_color}{step_text}: {self.ratio_color}{batch_number:d} "
        else:
            digits = int(math.log10(steps)) + 1
            if ratio:
                ratio_text = f"/{steps:d}"
            formatted_step = f"{self.text_color}{step_text}: {self.ratio_color}{batch_number:{digits}d}{ratio_text} "
        return formatted_step

    def _get_formatted_metrics(self, metrics_str: str) -> str:
        formatted_metrics = ""
        for metric in metrics_str.split(","):
            name_value = metric.split(":")
            name = name_value[0]
            value = name_value[1]
            formatted_metrics += self.text_color + name + ":" + self.metric_value_color + value

        return formatted_metrics

    def _batch_update(
        self, remaining_time: float, batch_number: int, metrics_str: str, steps: Union[int, None] = None
    ) -> str:
        update = ""
        if self.progress_bar:
            update += self._get_formatted_step(batch_number, steps)

            self.steps_progress_bar.update()

            update += str(self.steps_progress_bar) + self._get_formatted_time(remaining_time, steps)
        else:
            update += self._get_formatted_step(batch_number, steps) + self._get_formatted_time(remaining_time, steps)

        if metrics_str != "":
            update += self._get_formatted_metrics(metrics_str)

        if self.style_reset and not jupyter:
            # We skip it for Jupyter since the color token appear and otherwise the
            # traceback will be colored using a shell or else.
            update += Style.RESET_ALL

        return update

    def _do_update(self):
        if not jupyter:
            return True

        new_time = time.time()
        if self.prev_print_time is None or new_time - self.prev_print_time >= ColorProgress.JUPYTER_PRINT_RATE:
            self.prev_print_time = new_time
            return True

        return False

    def _pad_length(self, message):
        new_message_length = len(message)
        if new_message_length < self.prev_message_length:
            # Pad current message to overwrite the previous longuer message.
            message = message.ljust(self.prev_message_length)
        self.prev_message_length = new_message_length
        return message

    def _update_print(self, message: str) -> None:
        """
        Print a update message.
        """
        if self._do_update():
            message = self._pad_length(message)
            sys.stdout.write(message)
            sys.stdout.flush()

    def _end_print(self, message: str) -> None:
        """
        Print a update message but using print to create a new line after.
        """
        message = self._pad_length(message)
        print(message)
        sys.stdout.flush()
        self.prev_print_time = None
        self.prev_message_length = 0
