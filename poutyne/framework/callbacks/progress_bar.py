import math
from typing import Union


class ProgressBar:
    """
    ProgressBar class keeps an update of the iteration and output the corresponding update progress bar.

    Args:
          steps (int): The number of steps.
          bar_format (str, optional): User defined format for the bar_format. By default the setting is
            {percentage} |{bar}|. The two argument must be {percentage} and {bar}.
          bar_character (str): The bar character to use. (Default value = \u2588 which is 'block'.)

    Attributes:
        total_steps (int): The total steps to do.
        bar_format (str): The format of the bar.
        bar_character (str): The bar character.
        bar_len (int): The size of the bar.
        actual_steps (int): Number of steps done so far.
    """

    def __init__(self, steps: int, bar_format: Union[str, None] = None, bar_character: str = "\u2588") -> None:
        self.total_steps = steps

        if bar_format is not None:
            self.bar_format = bar_format
        else:
            self.bar_format = "{percentage} |{bar}"

        self.bar_character = bar_character
        self.bar_len = 20

        self.actual_steps = 0

    def reset(self) -> None:
        """
        Reset the counter of the progress bar.
        """
        self.actual_steps = 0

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
        percentage = f"{self.actual_steps / self.total_steps * 100:.2f}%"

        progress_bar = self.progress_bar_formatting()

        return self.bar_format.format(percentage=percentage, bar=progress_bar)

    def progress_bar_formatting(self) -> str:
        bar_len_complete = int(math.floor(self.actual_steps / self.total_steps * self.bar_len)) * self.bar_character
        bar_len_incomplete = (self.bar_len - len(bar_len_complete)) * " "
        progress_bar = bar_len_complete + bar_len_incomplete
        return progress_bar
