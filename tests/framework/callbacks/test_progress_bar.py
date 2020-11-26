from unittest import TestCase

from poutyne import ProgressBar

# bar size is by 25
default_bar_size = 25


class ProgressBarTest(TestCase):

    def test_default_bar_character_is_properly_step(self):
        smaller_bar_size = 24
        progress_bar = ProgressBar(steps=smaller_bar_size, bar_length=default_bar_size, bar_character="#")

        progress_bar.update()
        self.assertEqual(progress_bar.progress_bar_formatting().strip(), "#")

    def test_bar_formatting_steps_equal_than_bar_size(self):
        equal_bar_size = 25
        progress_bar = ProgressBar(steps=equal_bar_size, bar_length=default_bar_size, bar_character="#")

        step = 0
        while step < equal_bar_size:
            step += 1

            progress_bar.update()
            self.assertEqual(progress_bar.progress_bar_formatting(), "#" * step + " " * (equal_bar_size - step))

    def test_bar_formatting_steps_greater_than_bar_size(self):
        greater_bar_size = 50
        block_size = 2
        progress_bar = ProgressBar(steps=greater_bar_size, bar_length=default_bar_size, bar_character="#")

        step = 0
        while step < greater_bar_size:
            step += block_size

            progress_bar.update(block_size)
            number_blocks_bar = round(step / block_size)
            number_blocks_padding = (default_bar_size - number_blocks_bar)
            self.assertEqual(progress_bar.progress_bar_formatting(),
                             "#" * number_blocks_bar + " " * number_blocks_padding)

    def test_bar_formatting_steps_greaterodd_than_bar_size(self):
        smaller_bar_size = 111
        progress_bar = ProgressBar(steps=smaller_bar_size, bar_length=default_bar_size, bar_character="#")

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
        smaller_bar_size = 109
        bar_length = 20
        progress_bar = ProgressBar(steps=smaller_bar_size, bar_length=bar_length, bar_character="#")

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
        smaller_bar_size = 103
        bar_length = 19
        progress_bar = ProgressBar(steps=smaller_bar_size, bar_length=bar_length, bar_character="#")

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
        for steps in range(26, 200):
            progress_bar = ProgressBar(steps=steps, bar_length=default_bar_size, bar_character="#")
            progress_bar.update(steps)
            self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 25 + " " * 0)

    def test_one_before_last_step_is_only_24_bar(self):
        for steps in range(26, 200):
            progress_bar = ProgressBar(steps=steps, bar_length=default_bar_size, bar_character="#")
            progress_bar.update(steps - 1)
            self.assertEqual(progress_bar.progress_bar_formatting(), "#" * 24 + " " * 1)
