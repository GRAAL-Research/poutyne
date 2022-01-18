import io
import sys
from typing import List
from unittest import TestCase


class CaptureOutputBase(TestCase):
    def _capture_output(self) -> None:
        self.test_out = io.StringIO()
        self.original_output = sys.stdout
        sys.stdout = self.test_out

    def assertStdoutContains(self, values: List) -> None:
        for value in values:
            self.assertIn(value, self.test_out.getvalue().strip())
