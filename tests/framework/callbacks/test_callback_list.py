from unittest import TestCase
from unittest.mock import MagicMock, call

from poutyne import Callback, CallbackList, Model


class CallbackListTest(TestCase):
    def setUp(self) -> None:
        self.initial_callback = MagicMock(spec=Callback)
        self.callback_list = CallbackList([self.initial_callback])

    def test_append_callback(self):
        self.assertEqual(len(self.callback_list.callbacks), 1)
        a_callback = MagicMock(spec=Callback)
        self.callback_list.append(a_callback)

        self.assertEqual(len(self.callback_list.callbacks), 2)

    def test_set_params(self):
        params_dict = {"a_param": 1.0}
        self.callback_list.set_params(params_dict)
        self.initial_callback.assert_has_calls([call.set_params(params_dict)])

    def test_set_model(self):
        a_model = MagicMock(spec=Model)
        self.callback_list.set_model(a_model)
        self.initial_callback.assert_has_calls([call.set_model(a_model)])

    def test_iterator(self):
        a_callback = MagicMock(spec=Callback)
        self.callback_list.append(a_callback)
        self.assertEqual(len(list(self.callback_list)), 2)
