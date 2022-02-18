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

from unittest import TestCase

from poutyne import ProgressBar


class ProgressBarTest(TestCase):
    BAR_LENGTH = 25

    def test_default_bar_character_is_properly_step(self):
        smaller_bar_length = ProgressBarTest.BAR_LENGTH - 1
        progress_bar = ProgressBar(
            steps=smaller_bar_length,
            bar_length=ProgressBarTest.BAR_LENGTH,
            bar_character="#",
            partial_bar_characters=[" "],
        )

        progress_bar.update()
        self.assertEqual(progress_bar.progress_bar_formatting().strip(), "#")

    def test_bar_formatting_steps_equal_to_bar_length(self):
        equal_bar_length = ProgressBarTest.BAR_LENGTH
        progress_bar = ProgressBar(
            steps=equal_bar_length,
            bar_length=ProgressBarTest.BAR_LENGTH,
            bar_character="#",
            partial_bar_characters=[" "],
        )

        self.assertEqual(progress_bar.progress_bar_formatting(), " " * equal_bar_length)
        for step in range(1, equal_bar_length + 1):
            progress_bar.update()
            self.assertEqual(progress_bar.progress_bar_formatting(), "#" * step + " " * (equal_bar_length - step))

    def test_bar_formatting_steps_greater_than_bar_length(self):
        block_size = 2
        greater_bar_length = ProgressBarTest.BAR_LENGTH * 2
        progress_bar = ProgressBar(
            steps=greater_bar_length,
            bar_length=ProgressBarTest.BAR_LENGTH,
            bar_character="#",
            partial_bar_characters=[" "],
        )

        self.assertEqual(progress_bar.progress_bar_formatting(), " " * ProgressBarTest.BAR_LENGTH)
        for step in range(block_size, greater_bar_length + 1, block_size):
            progress_bar.update(block_size)
            number_blocks_bar = round(step / block_size)
            number_blocks_padding = ProgressBarTest.BAR_LENGTH - number_blocks_bar
            self.assertEqual(
                progress_bar.progress_bar_formatting(), "#" * number_blocks_bar + " " * number_blocks_padding
            )

    def test_bar_formatting_steps_greaterodd_than_bar_length(self):
        steps = 111
        bar_length = 25
        progress_bar = ProgressBar(steps=steps, bar_length=bar_length, bar_character="#", partial_bar_characters=[" "])

        self.assertEqual(progress_bar.progress_bar_formatting(), " " * 25)

        # block_size is equal to 111 / 25
        progress_bar.update(5)  # after 5 steps -> (5 // block_size)
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" + " " * 24)

        progress_bar.update(4)  # after 9
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 2 + " " * 23)

        progress_bar.update(98)  # after 107
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 24 + " " * 1)

        progress_bar.update(3)  # after 110
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 24 + " " * 1)

        progress_bar.update(1)  # after 111
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 25 + " " * 0)

    def test_other_bar_length(self):
        steps = 109
        bar_length = 20
        progress_bar = ProgressBar(steps=steps, bar_length=bar_length, bar_character="#", partial_bar_characters=[" "])

        self.assertEqual(progress_bar.progress_bar_formatting(), " " * 20)

        # block_size is equal to 109 / 20
        progress_bar.update(6)  # after 6 steps -> floor(6 / block_size)
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" + " " * 19)

        progress_bar.update(5)  # after 11
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 2 + " " * 18)

        progress_bar.update(93)  # after 104
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 19 + " " * 1)

        progress_bar.update(4)  # after 108
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 19 + " " * 1)

        progress_bar.update(1)  # after 109
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 20 + " " * 0)

    def test_odd_bar_length(self):
        steps = 103
        bar_length = 19
        progress_bar = ProgressBar(steps=steps, bar_length=bar_length, bar_character="#", partial_bar_characters=[" "])

        self.assertEqual(progress_bar.progress_bar_formatting(), " " * 19)

        # block_size is equal to 103 / 19
        progress_bar.update(6)  # after 6 steps -> floor(6 / block_size)
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" + " " * 18)

        progress_bar.update(5)  # after 11
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 2 + " " * 17)

        progress_bar.update(87)  # after 98
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 18 + " " * 1)

        progress_bar.update(4)  # after 102
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 18 + " " * 1)

        progress_bar.update(1)  # after 103
        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 19 + " " * 0)

    def test_last_step_is_correctly_formatted(self):
        for steps in range(1, 200):
            progress_bar = ProgressBar(
                steps=steps, bar_length=ProgressBarTest.BAR_LENGTH, bar_character="#", partial_bar_characters=[" "]
            )
            progress_bar.update(steps)
            self.assertEqual(progress_bar.progress_bar_formatting(), "#" * ProgressBarTest.BAR_LENGTH + " " * 0)

    def test_one_before_last_step_is_only_24_bar(self):
        for steps in range(ProgressBarTest.BAR_LENGTH + 1, 200):
            progress_bar = ProgressBar(
                steps=steps, bar_length=ProgressBarTest.BAR_LENGTH, bar_character="#", partial_bar_characters=[" "]
            )
            progress_bar.update(steps - 1)
            self.assertEqual(progress_bar.progress_bar_formatting(), "#" * (ProgressBarTest.BAR_LENGTH - 1) + " " * 1)

    def test_partial_bar_characters(self):
        steps = 80
        bar_length = 10
        num_partials = 8
        bar_characters = list(map(str, range(num_partials)))
        progress_bar = ProgressBar(
            steps=steps, bar_length=bar_length, bar_character="#", partial_bar_characters=bar_characters
        )
        for step in range(steps):
            partial_bar_character = bar_characters[step % num_partials]
            num_characters = step // num_partials
            self.assertEqual(len(progress_bar.progress_bar_formatting()), bar_length)
            self.assertEqual(
                progress_bar.progress_bar_formatting(),
                "#" * num_characters + partial_bar_character + " " * (bar_length - num_characters - 1),
            )
            progress_bar.update()

        self.assertEqual(progress_bar.progress_bar_formatting(), "#" * bar_length)
