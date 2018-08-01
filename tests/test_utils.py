import unittest
from unittest import TestCase
from unittest.mock import MagicMock, call

import torch

from pytoune import torch_apply

class TorchApplyTest(TestCase):
    def test_apply_on_list(self):
        my_list = [MagicMock(spec=torch.Tensor) for _ in range(10)]
        torch_apply(my_list, lambda t: t.cpu())
        self._test_method_calls(my_list)

    def test_apply_on_recursive_list(self):
        my_list = [MagicMock(spec=torch.Tensor) for _ in range(2)]
        my_list.append([MagicMock(spec=torch.Tensor) for _ in range(3)])
        my_list += [MagicMock(spec=torch.Tensor) for _ in range(1)]
        torch_apply(my_list, lambda t: t.cpu())
        self._test_method_calls(my_list[:2] + my_list[2] + my_list[3:])

    def test_apply_on_tuple(self):
        my_tuple = tuple(MagicMock(spec=torch.Tensor) for _ in range(10))
        torch_apply(my_tuple, lambda t: t.cpu())
        self._test_method_calls(my_tuple)

    def test_apply_on_recursive_tuple(self):
        my_tuple = tuple(MagicMock(spec=torch.Tensor) for _ in range(2))
        my_tuple += (tuple(MagicMock(spec=torch.Tensor) for _ in range(3)),)
        my_tuple += tuple(MagicMock(spec=torch.Tensor) for _ in range(1))
        torch_apply(my_tuple, lambda t: t.cpu())
        self._test_method_calls(my_tuple[:2] + my_tuple[2] + my_tuple[3:])

    def test_apply_on_dict(self):
        my_dict = {}
        for k in ['a', 'b', 'c']:
            my_dict[k] = MagicMock(spec=torch.Tensor)
        torch_apply(my_dict, lambda t: t.cpu())
        self._test_method_calls(list(my_dict.values()))

    def test_apply_on_recursive_dict(self):
        my_dict = {}
        my_dict['a'] = MagicMock(spec=torch.Tensor)
        my_dict['b'] = {}
        for k in ['c', 'd']:
            my_dict['b'][k] = MagicMock(spec=torch.Tensor)
        torch_apply(my_dict, lambda t: t.cpu())
        self._test_method_calls([my_dict['a'], *my_dict['b'].values()])

    def test_apply_on_recursive_data_structure(self):
        my_obj = {
            'a': [MagicMock(spec=torch.Tensor) for _ in range(3)],
            'b': tuple(MagicMock(spec=torch.Tensor) for _ in range(2)),
            'c': {
                'd': [MagicMock(spec=torch.Tensor) for _ in range(3)]
            },
            'e': MagicMock(spec=torch.Tensor)
        }
        torch_apply(my_obj, lambda t: t.cpu())
        self._test_method_calls(my_obj['a'] + list(my_obj['b']) + my_obj['c']['d'] + [my_obj['e']])

    def test_apply_on_object_with_no_tensor(self):
        my_obj = {
            'a': 5,
            'b': 3.141592,
            'c': {
                'd': [1, 2, 3]
            }
        }
        ret = torch_apply(my_obj, lambda t: t.cpu())
        self.assertEqual(ret, my_obj)
        self.assertFalse(ret is my_obj)

    def test_apply_with_replacement_to_no_tensor(self):
        my_obj = [MagicMock(spec=torch.Tensor)]
        ret = torch_apply(my_obj, lambda t: 123)
        self.assertEqual(ret, [123])

    def _test_method_calls(self, mock_list):
        print(mock_list)
        for mock in mock_list:
            self.assertEqual(mock.method_calls, [call.cpu()])

if __name__ == '__main__':
    unittest.main()
