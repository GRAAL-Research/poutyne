import sys
import warnings
from typing import Dict, Union

from poutyne.framework.callbacks.progress_bar import ProgressBar


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
        if shell == 'ZMQInteractiveShell':
            jupyter = True
        else:
            init()
            jupyter = False

    except NameError:
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
        self.epoch_formatted_text = ""

    def on_epoch_begin(self, epoch_number, epochs) -> None:
        if self.progress_bar:
            self.steps_progress_bar.reset()

        self._set_epoch_formatted_text(epoch_number, epochs)

    def on_train_batch_end(self,
                           remaining_time: float,
                           batch_number: int,
                           metrics_str: str,
                           steps: Union[int, None] = None) -> None:
        # pylint: disable=too-many-arguments
        """
        Format on train batch end for a steps the epoch ratio (so far / to do), the total time for the epoch, the steps
        done and the metrics name and values.
        """
        update = self.epoch_formatted_text

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

        sys.stdout.write(update)
        sys.stdout.flush()

    def on_epoch_end(self, epoch_total_time: float, steps: int, metrics_str: str) -> None:
        # pylint: disable=too-many-arguments
        """
        Format on epoch end the epoch ratio (so far / to do), the total time for the epoch, the steps done and the
        metrics name and values.
        """
        update = self.epoch_formatted_text

        if self.progress_bar:
            update += self._get_formatted_step(steps, steps)

            update += str(self.steps_progress_bar) + self._get_formatted_epoch_total_time(epoch_total_time)
        else:
            update += self._get_formatted_epoch_total_time(epoch_total_time) + self._get_formatted_step(steps, steps)
        update += self._get_formatted_metrics(metrics_str)

        if self.style_reset:
            update += Style.RESET_ALL

        print(update)
        sys.stdout.flush()

    def set_progress_bar(self, number_steps_per_epoch):
        self.steps_progress_bar = ProgressBar(number_steps_per_epoch,
                                              bar_format="%s{percentage} |%s{bar}%s|" %
                                              (self.text_color, self.progress_bar_color, self.text_color))
        self.progress_bar = True

    def _set_epoch_formatted_text(self, epoch_number: int, epochs: int) -> None:
        self.epoch_formatted_text = "\r" + self.text_color + "Epoch: " + self.ratio_color + "%d/%d " % (epoch_number,
                                                                                                        epochs)

    def _get_formatted_epoch_total_time(self, epoch_total_time: float) -> str:
        return self.time_color + "%.0fs " % epoch_total_time

    def _get_formatted_time(self, time: float, steps) -> str:
        if steps is None:
            formatted_time = self.time_color + "%.2fs/step " % time
        else:
            formatted_time = self.text_color + "ETA: " + self.time_color + "%.2fs " % time
        return formatted_time

    def _get_formatted_step(self, batch_number: int, steps: Union[int, None]) -> str:
        if steps is None:
            formatted_step = self.text_color + "Step: " + self.ratio_color + "%d " % batch_number
        else:
            formatted_step = self.text_color + "Step: " + self.ratio_color + "%d/%d " % (batch_number, steps)
        return formatted_step

    def _get_formatted_metrics(self, metrics_str: str) -> str:
        formatted_metrics = ""
        for metric in metrics_str.split(","):
            name_value = metric.split(":")
            name = name_value[0]
            value = name_value[1]
            formatted_metrics += self.text_color + name + ":" + self.metric_value_color + value

        return formatted_metrics
