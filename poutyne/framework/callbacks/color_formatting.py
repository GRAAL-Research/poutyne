import sys
import warnings
from typing import Dict, Union

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class EmptyStringAttrClass:
    """
    Class to emulate the Fore and Style class of colorama with a class that as an empty string for every attributes.
    """

    def __getattr__(self, attr):
        return ''


try:
    from colorama import Fore, Style, init

    colorama = True

    init()
except ImportError:
    colorama = None

    Fore = EmptyStringAttrClass()
    Style = EmptyStringAttrClass()

default_color_settings = {
    "text_color": 'LIGHTYELLOW_EX',
    "ratio_color": "LIGHTBLUE_EX",
    "metric_value_color": "LIGHTCYAN_EX",
    "time_color": "GREEN",
    "progress_bar_color": "LIGHTGREEN_EX"
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
            self.text_color = getattr(Fore, color_settings["text_color"])
            self.ratio_color = getattr(Fore, color_settings["ratio_color"])
            self.metric_value_color = getattr(Fore, color_settings["metric_value_color"])
            self.time_color = getattr(Fore, color_settings["time_color"])
            self.progress_bar_color = getattr(Fore, color_settings["progress_bar_color"])
        else:
            self.text_color = ""
            self.ratio_color = ""
            self.metric_value_color = ""
            self.time_color = ""
            self.progress_bar_color = ""

        self.progress_bar = False
        self.steps_progress_bar = None
        self.epoch_progress_bar = None

    def on_epoch_begin(self) -> None:
        if self.progress_bar:
            self.steps_progress_bar.reset()

    def on_epoch_end(self, epoch_number: int, epochs: int, epoch_total_time: float, steps: int,
                     metrics_str: str) -> None:
        # pylint: disable=too-many-arguments
        """
        Format on epoch end the epoch ratio (so far / to do), the total time for the epoch, the steps done and the
        metrics name and values.
        """
        sys.stdout.write(
            self._epoch_formatting(epoch_number, epochs) + self._step_formatting(steps, steps) +
            self._epoch_total_time_formatting(epoch_total_time) + self._metric_formatting(metrics_str) +
            Style.RESET_ALL)
        sys.stdout.flush()

        if self.progress_bar:
            self.epoch_progress_bar.update()

    def on_train_batch_end_steps(self, epoch_number: int, epochs: int, remaining_time: float, batch_number: int,
                                 steps: int, metrics_str: str) -> None:
        # pylint: disable=too-many-arguments
        """
        Format on train batch end for a steps the epoch ratio (so far / to do), the total time for the epoch, the steps
        done and the metrics name and values.
        """

        if self.progress_bar:
            self.steps_progress_bar.update()

        sys.stdout.write(
            self._epoch_formatting(epoch_number, epochs) + self._step_formatting(batch_number, steps) +
            self._ETA_formatting(remaining_time) + self._metric_formatting(metrics_str) + Style.RESET_ALL)

        sys.stdout.flush()

    def on_train_batch_end(self, epoch_number: int, epochs: int, times_mean: float, batch_number: int,
                           metrics_str: str) -> None:
        # pylint: disable=too-many-arguments
        """
        Format on train batch end the epoch ratio (so far / to do), the total time for the epoch, the steps
        done and the metrics name and values.
        """

        sys.stdout.write(
            self._epoch_formatting(epoch_number, epochs) + self._ETA_formatting(times_mean) +
            self._step_formatting(batch_number) + self._metric_formatting(metrics_str) + Style.RESET_ALL)
        sys.stdout.flush()

    def on_train_end(self):
        if self.progress_bar:
            self.steps_progress_bar.close()
            self.epoch_progress_bar.close()
            sys.stdout.flush()

    def set_progress_bar(self, number_steps_per_epoch, number_of_epoch):
        if colorama is None:
            warnings.warn("The tqdm package was not imported. Consider installing it for colorlog.", ImportWarning)
        else:
            self.steps_progress_bar = tqdm(
                total=number_steps_per_epoch,
                file=sys.stdout,
                dynamic_ncols=True,
                unit="step",
                bar_format="%s{l_bar}%s{bar}%s| %s{rate_fmt} %s" %
                (self.text_color, self.progress_bar_color, self.text_color, self.time_color, Style.RESET_ALL),
                leave=False)
            self.steps_progress_bar.clear()
            self.epoch_progress_bar = tqdm(total=number_of_epoch,
                                           file=sys.stdout,
                                           dynamic_ncols=True,
                                           unit="epoch",
                                           bar_format="%s{l_bar}%s{bar}%s| {rate_fmt}%s" %
                                           (self.text_color, self.progress_bar_color, self.time_color, Style.RESET_ALL),
                                           leave=True,
                                           desc="The training is at ")
            self.progress_bar = True

    def _epoch_formatting(self, epoch_number: int, epochs: int) -> str:
        return self.text_color + "\rEpoch: " + self.ratio_color + "%d/%d " % (epoch_number, epochs)

    def _epoch_total_time_formatting(self, epoch_total_time: float) -> str:
        if self.progress_bar:
            string_formatted = self.text_color + self.time_color + str(
                self.steps_progress_bar) + self.time_color + "%.2fs " % epoch_total_time
        else:
            string_formatted = self.time_color + "%.0fs " % epoch_total_time
        return string_formatted

    def _ETA_formatting(self, time: float) -> str:
        if self.progress_bar:
            string_formatted = self.text_color + self.time_color + str(
                self.steps_progress_bar) + self.text_color + " ETA " + self.time_color + "%.2fs " % time
        else:
            string_formatted = self.text_color + "ETA " + self.time_color + "%.0fs " % time
        return string_formatted

    def _step_formatting(self, batch_number: int, steps: Union[int, None] = None) -> str:
        if steps is None:
            formatted_step = self.text_color + "Step: " + self.ratio_color + "%d " % batch_number
        else:
            formatted_step = self.text_color + "Step: " + self.ratio_color + "%d/%d " % (batch_number, steps)
        return formatted_step

    def _metric_formatting(self, metrics_str: str) -> str:
        formatted_metrics = ""
        for metric in metrics_str.split(","):
            name_value = metric.split(":")
            name = name_value[0]
            value = name_value[1]
            formatted_metrics += self.text_color + name + ": " + self.metric_value_color + value + " "
        return formatted_metrics
