from statistics import mean
from time import time
from typing import Union


class ProgressBar:
    """
    Progress Bar class to keep an update of the iteration and output the corresponding update bar.

    Args:
          steps (int): The number of steps.
          unit (str): The unit time. (Default value = it)
          bar_format (Union[str, None], optional): User define format for the bar_format. By default the setting is
            {percentage}|{bar}| {rate}. The three argument can be {percentage}, {bar} and {rate}.
          bar_character (str): The bar character to use. (Default value = \u2588 which is a colored block)
    Attributes:
        total_steps (int): The total steps to do.
        unit (str): The unit of the rate.
        bar_format (str): The format of the bar.
        bar_character (str): The bar character.
        bar_len (int): The size of the bar.
        block (int): The size of a block for the progress bar.
        actual_steps (int): Number of steps done so far.
        mean_time (float): The mean time per step.
        last_time (float): The last time a step was done.
        mean_rate (float): The mean rate the step are done.

    """

    def __init__(self, steps: int, unit: str = "it", bar_format: Union[str, None] = None,
                 bar_character: str = "\u2588") -> None:
        self.total_steps = steps
        self.unit = unit

        if bar_format is not None:
            self.bar_format = bar_format
        else:
            self.bar_format = "{percentage}|{bar}| {rate}"

        self.bar_character = bar_character

        if self.total_steps > 25:
            self.bar_len = 25
        else:
            self.bar_len = self.total_steps
        self.block = self.total_steps / self.bar_len

        self.actual_steps = 0

        self.mean_time = 0.0
        self.last_time = 0.0
        self.mean_rate = 0.0

    def reset(self) -> None:
        """
        Reset the counter of the progress bar.
        """
        self.actual_steps = 0
        self.mean_time = 0.0
        self.last_time = 0.0
        self.mean_rate = 0.0

    def update(self, n: int = 1) -> None:
        """
        To update the progress bar.
        Args:
            Value of the update. (Default value = 1)
        """
        self.actual_steps += n

    def __str__(self):
        """
        To format the progress bar for an output.
        """
        percentage = "%.2f" % (round(self.actual_steps / self.total_steps * 100, 2)) + "%"

        progress_bar = self.progress_bar_formatting()

        actual_time = time()
        delta_t = actual_time - self.last_time
        self.last_time = actual_time
        self.mean_time = mean([self.mean_time, delta_t])
        self.mean_rate = mean([round(1 / self.mean_time, 2), self.mean_rate])
        rate = str(round(self.mean_rate, 2)) + self.unit + "/s"

        return self.bar_format.replace("{percentage}", percentage).replace("{bar}",
                                                                           progress_bar).replace("{rate}", rate)

    def progress_bar_formatting(self) -> str:
        if self.actual_steps == self.total_steps:
            # Rounding problem for the final step in some case and we would be sometime be stuck at 24.
            progress_bar = self.bar_len * self.bar_character
        else:
            bar_len_complete = round(self.actual_steps // self.block) * self.bar_character
            if self.actual_steps > self.total_steps and len(bar_len_complete) == self.bar_len:
                # Case where we have a block near 1 and a number of step close to 25.
                bar_len_incomplete = (self.bar_len - 1) * " "
                progress_bar = bar_len_complete + bar_len_incomplete
            else:
                bar_len_incomplete = (self.bar_len - len(bar_len_complete)) * " "
                progress_bar = bar_len_complete + bar_len_incomplete
        return progress_bar
