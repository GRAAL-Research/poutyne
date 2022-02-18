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

# pylint: disable=protected-access
from unittest import TestCase
from unittest.mock import patch

from poutyne import EmptyStringAttrClass, ColorProgress


class EmptyStringAttrClassTest(TestCase):
    def test_empty_string_att_class_get_attr_return_empty_string(self):
        actual = getattr(EmptyStringAttrClass(), "an_attribute")
        expected = ""
        self.assertEqual(actual, expected)


class ColorProgressTest(TestCase):
    @patch("poutyne.framework.color_formatting.colorama", None)
    def test_no_colorama_and_coloring_raise_warning(self):
        with self.assertWarns(ImportWarning):
            ColorProgress(coloring=True)

    def test_format_duration_with_days(self):
        # 2 days * 24 hours * 60 minutes * 60 seconds
        a_two_days_duration = 2 * 24 * 60 * 60
        color_progress = ColorProgress(coloring=False)
        actual = color_progress._format_duration(a_two_days_duration)
        expected = "2d0h0m0.00s"

        self.assertEqual(actual, expected)

        # 35 hours (1 day and 12 hours) + 30 minutes + 30 seconds
        a_day_and_12_hours_30_minutes_30_seconds_duration = 36 * 60 * 60 + 30 * 60 + 30
        color_progress = ColorProgress(coloring=False)
        actual = color_progress._format_duration(a_day_and_12_hours_30_minutes_30_seconds_duration)
        expected = "1d12h30m30.00s"

        self.assertEqual(actual, expected)

    @patch("poutyne.framework.color_formatting.jupyter", True)
    def test_do_update_when_jupyter_return_true(self):
        color_progress = ColorProgress(coloring=False)

        self.assertTrue(color_progress._do_update())

    def test_do_update_when_no_coloring_return_true(self):
        color_progress = ColorProgress(coloring=False)

        self.assertTrue(color_progress._do_update())

    @patch("poutyne.framework.color_formatting.jupyter", True)
    def test_do_update_with_jupyter_coloring_and_no_previous_timer_return_true(self):
        color_progress = ColorProgress(coloring=True)

        self.assertTrue(color_progress._do_update())

    @patch("poutyne.framework.color_formatting.jupyter", True)
    @patch("poutyne.framework.color_formatting.time")
    def test_do_update_with_jupyter_coloring_and_no_update_needed_return_false(self, time_mock):
        color_progress = ColorProgress(coloring=True)
        color_progress.prev_print_time = 1.0
        time_mock.time.return_value = 1.01

        self.assertFalse(color_progress._do_update())

    @patch("poutyne.framework.color_formatting.jupyter", True)
    @patch("poutyne.framework.color_formatting.time")
    def test_do_update_with_jupyter_coloring_and_an_update_needed_return_true(self, time_mock):
        color_progress = ColorProgress(coloring=True)
        color_progress.prev_print_time = 1.0
        time_mock.time.return_value = 1.2

        self.assertTrue(color_progress._do_update())
