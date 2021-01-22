import math
import sys
import warnings
from typing import Dict, Union, Callable

from .progress_bar import ProgressBar


class EmptyStringAttrClass:
    """
    Class to emulate the Fore and Style class of colorama with a class that as an empty string for every attributes.
    """

    def __getattr__(self, attr):
        return ''


try:
    from colorama import Fore, Style, init

    colorama = True

    try:
        # We don't init when Jupyter Notebook see issue https://github.com/jupyter/notebook/issues/2284
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell in ['ZMQInteractiveShell', 'Shell']:
            jupyter = True
        else:
            init()
            jupyter = False

    except ImportError:
        init()
        jupyter = False

except ImportError:
    colorama = None
    jupyter = False

    Fore = EmptyStringAttrClass()
    Style = EmptyStringAttrClass()

default_color_settings = {
    "text_color": 'MAGENTA',
    "ratio_color": "CYAN",
    "metric_value_color": "LIGHTBLUE_EX",
    "time_color": "GREEN",
    "progress_bar_color": "MAGENTA"
}


class ColorProgress:
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
                raise KeyError("The key(s) {} are not supported color attributes.".format(', '.join(invalid_keys)))

            color_settings.update(coloring)
        elif coloring:
            color_settings = default_color_settings

        if color_settings is not None:
            self.style_reset = True
            self.text_color = getattr(Fore, color_settings["text_color"])
            self.ratio_color = getattr(Fore, color_settings["ratio_color"])
            self.metric_value_color = getattr(Fore, color_settings["metric_value_color"])
            self.time_color = getattr(Fore, color_settings["time_color"])
            self.progress_bar_color = getattr(Fore, color_settings["progress_bar_color"])

        else:
            self.style_reset = False
            self.text_color = ""
            self.ratio_color = ""
            self.metric_value_color = ""
            self.time_color = ""
            self.progress_bar_color = ""

        self.progress_bar = False
        self.steps_progress_bar = None
        self.formatted_text = "\r"

    def on_epoch_begin(self) -> None:
        if self.progress_bar:
            self.steps_progress_bar.reset()

    def on_train_batch_end(self,
                           remaining_time: float,
                           batch_number: int,
                           metrics_str: str,
                           steps: Union[int, None] = None) -> None:
        """
        Format on train batch end for a steps the epoch ratio (so far / to do), the total time for the epoch, the steps
        done and the metrics name and values.
        """
        self._on_batch_end(remaining_time, batch_number, metrics_str, steps)

    def on_valid_batch_end(self,
                           remaining_time: float,
                           batch_number: int,
                           metrics_str: str,
                           steps: Union[int, None] = None) -> None:
        """
        Format on valid batch end for a steps the epoch ratio (so far / to do), the total time, the steps
        done and the metrics name and values.
        """
        self._on_batch_end(remaining_time, batch_number, metrics_str, steps)

    def on_test_batch_end(self,
                          remaining_time: float,
                          batch_number: int,
                          metrics_str: str,
                          steps: Union[int, None] = None) -> None:
        """
        Format on test batch end for a steps the epoch ratio (so far / to do), the total time, the steps
        done and the metrics name and values.
        """
        self._on_batch_end(remaining_time, batch_number, metrics_str, steps)

    def _on_batch_end(self,
                      remaining_time: float,
                      batch_number: int,
                      metrics_str: str,
                      steps: Union[int, None] = None) -> None:
        update = self.formatted_text

        update += self._batch_update(remaining_time, batch_number, metrics_str, steps)

        self._update_print(update)

    def on_epoch_end(self, epoch_total_time: float, steps: int, metrics_str: str) -> None:
        """
        Format on epoch end: the epoch ratio (so far / to do), the total time for the epoch, the steps done and the
        metrics name and values.
        """
        self._on_end(epoch_total_time, steps, metrics_str, self._end_print)

    def on_valid_end(self, valid_total_time: float, steps: int, metrics_str: str) -> None:
        """
        Format on valid end: the total time for the validation, the steps done and the metrics name and values.
        """
        self._on_end(valid_total_time, steps, metrics_str, self._update_print)

    def on_test_end(self, test_total_time: float, steps: int, metrics_str: str) -> None:
        """
        Format on test end: the total time for the test, the steps done and the metrics name and values.
        """
        self._on_end(test_total_time, steps, metrics_str, self._end_print)

    def _on_end(self, total_time: float, steps: int, metrics_str: str, update_func: Callable) -> None:
        """
        Format on end: the total time for the loop, the steps done and the metrics name and values.
        """
        update = self.end_update(self.formatted_text, steps, total_time, metrics_str)

        update_func(update)

    def set_progress_bar(self, number_steps_per_epoch: int) -> None:
        bar_format = f"{self.text_color}{{percentage}} |{self.progress_bar_color}{{bar}}{self.text_color}|"
        self.steps_progress_bar = ProgressBar(number_steps_per_epoch, bar_format=bar_format)
        self.progress_bar = True

    def end_update(self, update: str, steps: int, total_time: float, metrics_str: str) -> str:
        if self.progress_bar:
            update += self._get_formatted_step(steps, steps)

            update += str(self.steps_progress_bar) + self._get_formatted_total_time(total_time)
        else:
            update += self._get_formatted_total_time(total_time) + self._get_formatted_step(steps, steps)
        update += self._get_formatted_metrics(metrics_str)

        if self.style_reset:
            update += Style.RESET_ALL

        return update

    def _get_formatted_total_time(self, total_time: float) -> str:
        return f"{self.time_color}{total_time:.2f}s "

    def _get_formatted_time(self, time: float, steps: Union[int, None]) -> str:
        if steps is None:
            formatted_time = f"{self.time_color}{time:.2f}s/step "
        else:
            formatted_time = f"{self.text_color}ETA: {self.time_color}{time:.2f}s "
        return formatted_time

    def _get_formatted_step(self, batch_number: int, steps: Union[int, None]) -> str:
        if steps is None:
            formatted_step = f"{self.text_color}Step: {self.ratio_color}{batch_number:d} "
        else:
            digits = int(math.log10(steps)) + 1
            formatted_step = f"{self.text_color}Step: {self.ratio_color}{batch_number:{digits}d}/{steps:d} "
        return formatted_step

    def _get_formatted_metrics(self, metrics_str: str) -> str:
        formatted_metrics = ""
        for metric in metrics_str.split(","):
            name_value = metric.split(":")
            name = name_value[0]
            value = name_value[1]
            formatted_metrics += self.text_color + name + ":" + self.metric_value_color + value

        return formatted_metrics

    def _batch_update(self,
                      remaining_time: float,
                      batch_number: int,
                      metrics_str: str,
                      steps: Union[int, None] = None) -> str:
        update = ""
        if self.progress_bar:
            update += self._get_formatted_step(batch_number, steps)

            self.steps_progress_bar.update()

            update += str(self.steps_progress_bar) + self._get_formatted_time(remaining_time, steps)
        else:
            update += self._get_formatted_time(remaining_time, steps) + self._get_formatted_step(batch_number, steps)

        update += self._get_formatted_metrics(metrics_str)

        if self.style_reset and not jupyter:
            # We skip it for Jupyter since the color token appear and otherwise the
            # traceback will be colored using a shell or else.
            update += Style.RESET_ALL

        return update

    @staticmethod
    def _update_print(message: str) -> None:
        """
        Print a update message.
        """
        sys.stdout.write(message)
        sys.stdout.flush()

    @staticmethod
    def _end_print(message: str) -> None:
        """
        Print a update message but using print to create a new line after.
        """
        print(message)
        sys.stdout.flush()
