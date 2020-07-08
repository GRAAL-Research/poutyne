from unittest import TestCase

from poutyne.framework import ProgressBar

# bar size is by 25
default_bar_size = 25


class ProgressBarTest(TestCase):
    def test_default_bar_character_is_properly_step(self):
        smaller_bar_size = 24
        bar = ProgressBar(steps=smaller_bar_size, bar_character="#")

        bar.update()
        self.assertEqual(bar.progress_bar_formatting().strip(), "#")

    def test_bar_formatting_steps_equal_than_bar_size(self):
        equal_bar_size = 25
        bar = ProgressBar(steps=equal_bar_size, bar_character="#")

        step = 0
        while step < equal_bar_size:
            step += 1

            bar.update()
            self.assertEqual(bar.progress_bar_formatting(), "#" * step + " " * (equal_bar_size - step))

    def test_bar_formatting_steps_greater_than_bar_size(self):
        greater_bar_size = 50
        block_size = 2
        bar = ProgressBar(steps=greater_bar_size, bar_character="#")

        step = 0
        while step < greater_bar_size:
            step += block_size

            bar.update(block_size)
            number_blocks_bar = round(step / block_size)
            number_blocks_padding = (default_bar_size - number_blocks_bar)
            self.assertEqual(bar.progress_bar_formatting(), "#" * number_blocks_bar + " " * number_blocks_padding)

    def test_bar_formatting_steps_greaterodd_than_bar_size(self):
        smaller_bar_size = 111
        bar = ProgressBar(steps=smaller_bar_size, bar_character="#")

        # block_size is equal to 111 / 25
        bar.update(5)  # after 5 steps -> (5 // block_size)
        self.assertEqual(bar.progress_bar_formatting(), "#" + " " * 24)

        bar.update(4)  # after 9
        self.assertEqual(bar.progress_bar_formatting(), "#" * 2 + " " * 23)

        bar.update(98)  # after 107
        self.assertEqual(bar.progress_bar_formatting(), "#" * 24 + " " * 1)

        bar.update(3)  # after 110
        self.assertEqual(bar.progress_bar_formatting(), "#" * 24 + " " * 1)

        bar.update(1)  # after 111
        self.assertEqual(bar.progress_bar_formatting(), "#" * 25 + " " * 0)

    def test_last_step_is_correctly_formatted(self):
        for steps in range(26, 200):
            bar = ProgressBar(steps=steps, bar_character="#")
            bar.update(steps)
            self.assertEqual(bar.progress_bar_formatting(), "#" * 25 + " " * 0)

    def test_one_before_last_step_is_only_24_bar(self):
        for steps in range(26, 200):
            bar = ProgressBar(steps=steps, bar_character="#")
            bar.update(steps - 1)
            self.assertEqual(bar.progress_bar_formatting(), "#" * 24 + " " * 1)
