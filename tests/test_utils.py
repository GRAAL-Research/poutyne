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

# -*- coding: utf-8 -*-
import warnings
from collections import OrderedDict

import unittest
from unittest import TestCase, skipIf
from unittest.mock import MagicMock, call

import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence

from poutyne import TensorDataset, torch_apply, warning_settings, get_batch_size
from poutyne.utils import _concat
from tests.utils import populate_packed_sequence


class TorchApplyTest(TestCase):
    cpu_call = call.cpu()
    device = "cuda:0"
    gpu_call = call.to(device)

    def test_apply_on_list(self):
        my_list = [MagicMock(spec=torch.Tensor) for _ in range(10)]
        torch_apply(my_list, lambda t: t.cpu())
        self._test_method_calls(my_list, device_call=self.cpu_call)

    def test_apply_on_recursive_list(self):
        my_list = [MagicMock(spec=torch.Tensor) for _ in range(2)]
        my_list.append([MagicMock(spec=torch.Tensor) for _ in range(3)])
        my_list += [MagicMock(spec=torch.Tensor) for _ in range(1)]
        torch_apply(my_list, lambda t: t.cpu())
        self._test_method_calls(my_list[:2] + my_list[2] + my_list[3:], device_call=self.cpu_call)

    def test_apply_on_tuple(self):
        my_tuple = tuple(MagicMock(spec=torch.Tensor) for _ in range(10))
        torch_apply(my_tuple, lambda t: t.cpu())
        self._test_method_calls(my_tuple, device_call=self.cpu_call)

    def test_apply_on_recursive_tuple(self):
        my_tuple = tuple(MagicMock(spec=torch.Tensor) for _ in range(2))
        my_tuple += (tuple(MagicMock(spec=torch.Tensor) for _ in range(3)),)
        my_tuple += tuple(MagicMock(spec=torch.Tensor) for _ in range(1))
        torch_apply(my_tuple, lambda t: t.cpu())
        self._test_method_calls(my_tuple[:2] + my_tuple[2] + my_tuple[3:], device_call=self.cpu_call)

    def test_apply_on_dict(self):
        my_dict = {}
        for k in ['a', 'b', 'c']:
            my_dict[k] = MagicMock(spec=torch.Tensor)
        torch_apply(my_dict, lambda t: t.cpu())
        self._test_method_calls(list(my_dict.values()), device_call=self.cpu_call)

    def test_apply_on_recursive_dict(self):
        my_dict = {}
        my_dict['a'] = MagicMock(spec=torch.Tensor)
        my_dict['b'] = {}
        for k in ['c', 'd']:
            my_dict['b'][k] = MagicMock(spec=torch.Tensor)
        torch_apply(my_dict, lambda t: t.cpu())
        self._test_method_calls([my_dict['a'], *my_dict['b'].values()], device_call=self.cpu_call)

    def test_apply_on_recursive_data_structure(self):
        my_obj = {
            'a': [MagicMock(spec=torch.Tensor) for _ in range(3)],
            'b': tuple(MagicMock(spec=torch.Tensor) for _ in range(2)),
            'c': {'d': [MagicMock(spec=torch.Tensor) for _ in range(3)]},
            'e': MagicMock(spec=torch.Tensor),
        }
        torch_apply(my_obj, lambda t: t.cpu())
        self._test_method_calls(
            my_obj['a'] + list(my_obj['b']) + my_obj['c']['d'] + [my_obj['e']], device_call=self.cpu_call
        )

    def test_apply_on_object_with_no_tensor(self):
        my_obj = {'a': 5, 'b': 3.141592, 'c': {'d': [1, 2, 3]}}
        ret = torch_apply(my_obj, lambda t: t.cpu())
        self.assertEqual(ret, my_obj)
        self.assertFalse(ret is my_obj)

    def test_apply_with_replacement_to_no_tensor(self):
        my_obj = [MagicMock(spec=torch.Tensor)]
        ret = torch_apply(my_obj, lambda t: 123)
        self.assertEqual(ret, [123])

    def test_apply_with_packed_sequence(self):
        my_obj, data_mock, batch_sizes_mock = _setup_packed_sequence_obj_mock()
        torch_apply(my_obj, lambda t: t.cpu())

        self._test_method_calls([data_mock], device_call=self.cpu_call)
        self._test_not_in_method_calls([batch_sizes_mock], device_call=self.cpu_call)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_apply_with_packed_sequence_gpu(self):
        my_obj, data_mock, batch_sizes_mock = _setup_packed_sequence_obj_mock()
        torch_apply(my_obj, lambda t: t.to(self.device))

        self._test_method_calls([data_mock], device_call=self.gpu_call)
        self._test_not_in_method_calls([batch_sizes_mock], device_call=self.gpu_call)

    def test_apply_with_packed_sequence_integration(self):
        device = torch.device("cpu")
        pack_padded_sequences_vectors = populate_packed_sequence()
        process_packed_sequence = torch_apply(pack_padded_sequences_vectors, lambda t: t.to(device))
        self.assertTrue(isinstance(process_packed_sequence, PackedSequence))
        self.assertEqual(process_packed_sequence.data.device, device)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_apply_with_packed_sequence_integration_gpu(self):
        device = torch.device("cuda:0")
        pack_padded_sequences_vectors = populate_packed_sequence()
        process_packed_sequence = torch_apply(pack_padded_sequences_vectors, lambda t: t.to(device))
        self.assertTrue(isinstance(process_packed_sequence, PackedSequence))
        self.assertEqual(process_packed_sequence.data.device, device)

    def test_apply_with_tuple_with_an_packed_sequence(self):
        my_obj, data_mock, batch_sizes_mock = _setup_packed_sequence_obj_mock()
        torch_apply(my_obj, lambda t: t.cpu())

        self._test_method_calls([data_mock], device_call=self.cpu_call)
        self._test_not_in_method_calls([batch_sizes_mock], device_call=self.cpu_call)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_apply_with_tuple_with_an_packed_sequence_gpu(self):
        my_obj, data_mock, batch_sizes_mock = _setup_packed_sequence_obj_mock()
        torch_apply(my_obj, lambda t: t.to(self.device))

        self._test_method_calls([data_mock], device_call=self.gpu_call)
        self._test_not_in_method_calls([batch_sizes_mock], device_call=self.gpu_call)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_apply_with_tuple_with_an_packed_sequence_integration_gpu(self):
        device = torch.device("cuda:0")
        pack_padded_sequences_vectors = populate_packed_sequence()
        tupled_obj = (pack_padded_sequences_vectors, MagicMock())
        process_packed_sequence, _ = torch_apply(tupled_obj, lambda t: t.to(device))
        self.assertTrue(isinstance(process_packed_sequence, PackedSequence))
        self.assertEqual(process_packed_sequence.data.device, device)

    def _test_method_calls(self, mock_list, device_call):
        self.assertGreater(len(mock_list), 0)
        for mock in mock_list:
            self.assertEqual(mock.method_calls, [device_call])

    def _test_not_in_method_calls(self, mock_list, device_call):
        self.assertGreater(len(mock_list), 0)
        for mock in mock_list:
            self.assertNotIn(device_call, mock.method_calls)


def _setup_packed_sequence_obj_mock():
    data_mock = MagicMock(spec=torch.Tensor)
    batch_sizes_mock = MagicMock(spec=torch.Tensor)
    batch_sizes_mock.device.type = "cpu"

    my_obj = PackedSequence(data=data_mock, batch_sizes=batch_sizes_mock)
    return my_obj, data_mock, batch_sizes_mock


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
        self.assertEqual(concat.shape, (25,))

    def test_tuple_1(self):
        """
        Test the concatenation of a [([], [])]
        """
        obj = [(np.arange(5), np.ones(5) * 2)] * 5
        concat = _concat(obj)
        self.assertEqual(concat[0].shape, (25,))
        self.assertEqual(concat[1].shape, (25,))
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
        self.assertEqual(concat[0].shape, (25,))
        self.assertEqual(concat[1][0].shape, (25,))
        self.assertEqual(concat[1][1].shape, (25,))
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
        self.assertEqual(concat[0][0].shape, (25,))
        self.assertEqual(concat[0][1].shape, (25,))
        self.assertEqual(concat[1][0].shape, (25,))
        self.assertEqual(concat[1][1].shape, (25,))
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
        self.assertEqual(concat[0].shape, (25,))
        self.assertEqual(concat[1].shape, (25,))
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
        self.assertEqual(concat[0].shape, (25,))
        self.assertEqual(concat[1][0].shape, (25,))
        self.assertEqual(concat[1][1].shape, (25,))
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
        self.assertEqual(concat[0][0].shape, (25,))
        self.assertEqual(concat[0][1].shape, (25,))
        self.assertEqual(concat[1][0].shape, (25,))
        self.assertEqual(concat[1][1].shape, (25,))
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
        self.assertEqual(concat['a'].shape, (25,))
        self.assertEqual(concat['b'].shape, (25,))
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
        self.assertEqual(concat['a'][0].shape, (25,))
        self.assertEqual(concat['a'][1].shape, (25,))
        self.assertEqual(concat['b'].shape, (25,))

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
        self.assertEqual(concat['a'][0].shape, (25,))
        self.assertEqual(concat['a'][1].shape, (25,))
        self.assertEqual(concat['a'][2], (2,) * 5)
        self.assertEqual(concat['b'], [3] * 5)
        self.assertEqual(concat['c'], [4] * 5)

        for i in range(5):
            for j in range(5):
                self.assertTrue(concat['a'][0][i * 5 + j] == j)
        self.assertTrue((concat['a'][1] == 1).all())


class GetBatchSizeTest(TestCase):
    batch_size = 20

    def test_get_batch_size(self):
        batch_size = GetBatchSizeTest.batch_size
        x = np.random.rand(batch_size, 1).astype(np.float32)
        y = np.random.rand(batch_size, 1).astype(np.float32)

        batch_size2 = batch_size + 1
        x2 = np.random.rand(batch_size2, 1).astype(np.float32)
        y2 = np.random.rand(batch_size2, 1).astype(np.float32)

        other_batch_size = batch_size2 + 1

        inf_batch_size = get_batch_size(x, y)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = get_batch_size(x2, y2)
        self.assertEqual(inf_batch_size, batch_size2)

        inf_batch_size = get_batch_size(x, y2)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = get_batch_size(x2, y)
        self.assertEqual(inf_batch_size, batch_size2)

        inf_batch_size = get_batch_size((x, x2), y)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = get_batch_size((x2, x), y)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = get_batch_size((x, x2), (y, y2))
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = get_batch_size((x2, x), (y, y2))
        self.assertEqual(inf_batch_size, batch_size2)

        inf_batch_size = get_batch_size([x, x2], y)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = get_batch_size([x2, x], y)
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = get_batch_size([x, x2], [y, y2])
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = get_batch_size([x2, x], [y, y2])
        self.assertEqual(inf_batch_size, batch_size2)

        inf_batch_size = get_batch_size({'batch_size': other_batch_size, 'x': x}, {'y': y})
        self.assertEqual(inf_batch_size, other_batch_size)

        inf_batch_size = get_batch_size({'x': x}, {'batch_size': other_batch_size, 'y': y})
        self.assertEqual(inf_batch_size, other_batch_size)

        inf_batch_size = get_batch_size({'x': x}, {'y': y})
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = get_batch_size(OrderedDict([('x1', x), ('x2', x2)]), {'y': y})
        self.assertEqual(inf_batch_size, batch_size)

        inf_batch_size = get_batch_size(OrderedDict([('x1', x2), ('x2', x)]), {'y': y})
        self.assertEqual(inf_batch_size, batch_size2)

        inf_batch_size = get_batch_size([1, 2, 3], {'y': y})
        self.assertEqual(inf_batch_size, batch_size)

    def test_get_batch_size_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inf_batch_size = get_batch_size([1, 2, 3], [4, 5, 6])
            self.assertEqual(inf_batch_size, 1)
            self.assertEqual(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warning_settings['batch_size'] = 'ignore'
            inf_batch_size = get_batch_size([1, 2, 3], [4, 5, 6])
            self.assertEqual(inf_batch_size, 1)
            self.assertEqual(len(w), 0)


if __name__ == '__main__':
    unittest.main()
