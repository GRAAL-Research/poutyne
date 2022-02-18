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
from typing import Optional

BAR_EIGHTHS_CHARACTERS = [" ", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589"]


class ProgressBar:
    """
    ProgressBar class keeps an update of the iteration and output the corresponding update progress bar.

    Args:
        steps (int): The number of steps.
        bar_format (str, optional): User defined format for the bar_format. By default the setting is
            {percentage} |{bar}|. The two argument must be {percentage} and {bar}.
        bar_character (str): The bar character to use. (Default value = \u2588 which is 'block'.)
        partial_bar_characters (str): A list of character to use when a fraction of the bar character
            could be used. (Default value =
            ``[" ", "\u258f", "\u258e", "\u258d", "\u258c", "\u258b", "\u258a", "\u2589"]``
            that is from " " through 1/8th to 7/8th blocks)

    Attributes:
        total_steps (int): The total steps to do.
        bar_format (str): The format of the bar.
        bar_character (str): The bar character.
        bar_length (int): The size of the bar.
        actual_steps (int): Number of steps done so far.
    """

    def __init__(
        self,
        steps: int,
        *,
        bar_length: int = 20,
        bar_format: Optional[str] = None,
        bar_character: str = "\u2588",
        partial_bar_characters: Optional[list] = None,
    ) -> None:
        self.total_steps = steps

        if bar_format is not None:
            self.bar_format = bar_format
        else:
            self.bar_format = "{percentage} |{bar}"

        self.bar_character = bar_character
        self.partial_bar_characters = BAR_EIGHTHS_CHARACTERS
        if partial_bar_characters is not None:
            self.partial_bar_characters = partial_bar_characters
        self.bar_length = bar_length

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
        percentage = f"{self.actual_steps / self.total_steps * 100:6.2f}%"

        progress_bar = self.progress_bar_formatting()

        return self.bar_format.format(percentage=percentage, bar=progress_bar)

    def progress_bar_formatting(self) -> str:
        percentage = self.actual_steps / self.total_steps
        bar_length_complete = int(math.floor(percentage * self.bar_length)) * self.bar_character
        bar_length_incomplete = (self.bar_length - len(bar_length_complete) - 1) * " "

        partial_character = ""
        if percentage < 1.0:
            partial_percentage = percentage * self.bar_length - len(bar_length_complete)
            partial_index = int(math.floor(partial_percentage * len(self.partial_bar_characters)))
            partial_character = self.partial_bar_characters[partial_index]

        progress_bar = bar_length_complete + partial_character + bar_length_incomplete
        return progress_bar
