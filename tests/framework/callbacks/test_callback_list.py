from unittest import TestCase
from unittest.mock import MagicMock

from poutyne import Callback, CallbackList


class CallbackListTest(TestCase):
    def setUp(self) -> None:
        self.initial_callback = MagicMock(spec=Callback)
        self.callback_list = CallbackList([self.initial_callback])

    def test_append_callback(self):
        self.assertEqual(len(self.callback_list.callbacks), 1)
        a_callback = MagicMock(spec=Callback)
        self.callback_list.append(a_callback)

        self.assertEqual(len(self.callback_list.callbacks), 2)

    def test_iterator(self):
        a_callback = MagicMock(spec=Callback)
        self.callback_list.append(a_callback)
        self.assertEqual(len(list(self.callback_list)), 2)
