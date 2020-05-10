# -*- coding: utf-8 -*-
from unittest import TestCase
from unittest.mock import MagicMock, call
import unittest
import torch
import numpy as np
from poutyne.utils import TensorDataset, _concat
from poutyne import torch_apply


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
        my_tuple += (tuple(MagicMock(spec=torch.Tensor) for _ in range(3)), )
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
        my_obj = {'a': 5, 'b': 3.141592, 'c': {'d': [1, 2, 3]}}
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


class TensorDatasetTest(TestCase):
    def test_one_tensor(self):
        range20 = np.expand_dims(np.arange(20), 1)
        dataset = TensorDataset(range20)
        self.assertEqual(len(dataset), 20)
        for i in range(20):
            self.assertEqual(dataset[i], np.array([i]))

    def test_multiple_tensors(self):
        range20 = np.expand_dims(np.arange(20), 1)
        dataset = TensorDataset(range20, range20 * 2, range20 * 3)
        self.assertEqual(len(dataset), 20)
        self.assertEqual(type(dataset[0]), tuple)
        for i in range(20):
            self.assertEqual(dataset[i][0], i)
            self.assertEqual(dataset[i][1], i * 2)
            self.assertEqual(dataset[i][2], i * 3)

    def test_list_of_tensors(self):
        range20 = np.expand_dims(np.arange(20), 1)
        dataset = TensorDataset((range20, range20 * 2), range20 * 3)
        self.assertEqual(len(dataset), 20)
        self.assertEqual(type(dataset[0]), tuple)
        self.assertEqual(type(dataset[0][0]), tuple)
        self.assertEqual(type(dataset[0][-1]), np.ndarray)
        for i in range(20):
            self.assertEqual(dataset[i][0][0], i)
            self.assertEqual(dataset[i][0][1], i * 2)
            self.assertEqual(dataset[i][1], i * 3)

        dataset = TensorDataset((range20, range20 * 2), (range20 * 3, range20 * 4))
        self.assertEqual(len(dataset), 20)

        self.assertEqual(type(dataset[0]), tuple)
        self.assertEqual(type(dataset[1]), tuple)
        self.assertEqual(type(dataset[0][0]), tuple)
        self.assertEqual(type(dataset[0][1]), tuple)
        for i in range(20):
            self.assertEqual(type(dataset[i][0][0]), np.ndarray)
            self.assertEqual(type(dataset[i][0][1]), np.ndarray)
            self.assertEqual(dataset[i][0][0], i)
            self.assertEqual(dataset[i][0][1], i * 2)
            self.assertEqual(type(dataset[i][1][0]), np.ndarray)
            self.assertEqual(type(dataset[i][1][1]), np.ndarray)
            self.assertEqual(dataset[i][1][0], i * 3)
            self.assertEqual(dataset[i][1][1], i * 4)


class ConcatTest(TestCase):
    def test_single_array(self):
        """
        Test the concatenation of a single array
        """
        obj = [np.arange(5)] * 5
        concat = _concat(obj)
        self.assertEqual(concat.shape, (25, ))

    def test_tuple_1(self):
        """
        Test the concatenation of a [([], [])]
        """
        obj = [(np.arange(5), np.ones(5) * 2)] * 5
        concat = _concat(obj)
        self.assertEqual(concat[0].shape, (25, ))
        self.assertEqual(concat[1].shape, (25, ))
        for i in range(5):
            for j in range(5):
                self.assertTrue(concat[0][i * 5 + j] == j)
        self.assertTrue((concat[1] == 2).all())

    def test_tuple_2(self):
        """
        Test the concatenation of a [([], ([], []))]
        """
        obj = [(np.arange(5), (np.ones(5) * 2, np.ones(5) * 3))] * 5
        concat = _concat(obj)
        self.assertEqual(concat[0].shape, (25, ))
        self.assertEqual(concat[1][0].shape, (25, ))
        self.assertEqual(concat[1][1].shape, (25, ))
        for i in range(5):
            for j in range(5):
                self.assertTrue(concat[0][i * 5 + j] == j)
        self.assertTrue((concat[1][0] == 2).all())
        self.assertTrue((concat[1][1] == 3).all())

    def test_tuple_3(self):
        """
        Test the concatenation of a [(([], []), ([], []))]
        """
        obj = [((np.arange(5), np.ones(5)), (np.ones(5) * 2, np.ones(5) * 3))] * 5
        concat = _concat(obj)
        self.assertEqual(concat[0][0].shape, (25, ))
        self.assertEqual(concat[0][1].shape, (25, ))
        self.assertEqual(concat[1][0].shape, (25, ))
        self.assertEqual(concat[1][1].shape, (25, ))
        for i in range(5):
            for j in range(5):
                self.assertTrue(concat[0][0][i * 5 + j] == j)
        self.assertTrue((concat[0][1] == 1).all())
        self.assertTrue((concat[1][0] == 2).all())
        self.assertTrue((concat[1][1] == 3).all())

    def test_array_1(self):
        """
        Test the concatenation of a [[[], []]]
        """
        obj = [[np.arange(5), np.ones(5) * 2]] * 5
        concat = _concat(obj)
        self.assertEqual(concat[0].shape, (25, ))
        self.assertEqual(concat[1].shape, (25, ))
        for i in range(5):
            for j in range(5):
                self.assertTrue(concat[0][i * 5 + j] == j)
        self.assertTrue((concat[1] == 2).all())

    def test_array_2(self):
        """
        Test the concatenation of a [[[], ([], [])]]
        """
        obj = [[np.arange(5), [np.ones(5) * 2, np.ones(5) * 3]]] * 5
        concat = _concat(obj)
        self.assertEqual(concat[0].shape, (25, ))
        self.assertEqual(concat[1][0].shape, (25, ))
        self.assertEqual(concat[1][1].shape, (25, ))
        for i in range(5):
            for j in range(5):
                self.assertTrue(concat[0][i * 5 + j] == j)
        self.assertTrue((concat[1][0] == 2).all())
        self.assertTrue((concat[1][1] == 3).all())

    def test_array_3(self):
        """
        Test the concatenation of a [[[[], []], [[], []]]]
        """
        obj = [[[np.arange(5), np.ones(5)], [np.ones(5) * 2, np.ones(5) * 3]]] * 5
        concat = _concat(obj)
        self.assertEqual(concat[0][0].shape, (25, ))
        self.assertEqual(concat[0][1].shape, (25, ))
        self.assertEqual(concat[1][0].shape, (25, ))
        self.assertEqual(concat[1][1].shape, (25, ))
        for i in range(5):
            for j in range(5):
                self.assertTrue(concat[0][0][i * 5 + j] == j)
        self.assertTrue((concat[0][1] == 1).all())
        self.assertTrue((concat[1][0] == 2).all())
        self.assertTrue((concat[1][1] == 3).all())

    def test_dict_1(self):
        """
        Test list of dictionaries
        """
        obj = [{'a': np.arange(5), 'b': np.ones(5) * 2}] * 5
        concat = _concat(obj)
        self.assertEqual(concat['a'].shape, (25, ))
        self.assertEqual(concat['b'].shape, (25, ))
        for i in range(5):
            for j in range(5):
                self.assertTrue(concat['a'][i * 5 + j] == j)
        self.assertTrue((concat['b'] == 2).all())

    def test_dict_2(self):
        """
        Test list of dictionaries
        """
        obj = [{'a': (np.arange(5), np.ones(5)), 'b': np.ones(5) * 2}] * 5
        concat = _concat(obj)
        self.assertEqual(concat['a'][0].shape, (25, ))
        self.assertEqual(concat['a'][1].shape, (25, ))
        self.assertEqual(concat['b'].shape, (25, ))

        for i in range(5):
            for j in range(5):
                self.assertTrue(concat['a'][0][i * 5 + j] == j)
        self.assertTrue((concat['a'][1] == 1).all())
        self.assertTrue((concat['b'] == 2).all())

    def test_non_concatenable_values(self):
        obj = [3] * 5
        concat = _concat(obj)
        self.assertEqual(concat, obj)
        self.assertEqual(type(concat), type(obj))

    def test_non_concatenable_values2(self):
        obj = [{'a': (np.arange(5), np.ones(5), 2), 'b': 3, 'c': np.array(4)}] * 5
        concat = _concat(obj)
        self.assertEqual(concat['a'][0].shape, (25, ))
        self.assertEqual(concat['a'][1].shape, (25, ))
        self.assertEqual(concat['a'][2], (2, ) * 5)
        self.assertEqual(concat['b'], [3] * 5)
        self.assertEqual(concat['c'], [4] * 5)

        for i in range(5):
            for j in range(5):
                self.assertTrue(concat['a'][0][i * 5 + j] == j)
        self.assertTrue((concat['a'][1] == 1).all())


if __name__ == '__main__':
    unittest.main()
