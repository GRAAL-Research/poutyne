from unittest import TestCase, skipIf

import numpy as np
import torch

from poutyne import Accuracy, BinaryAccuracy, acc, bin_acc


class AccuracyTest(TestCase):

    def setUp(self):
        self.predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0],
                                         [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.5, 0.1, 0.2, 0.0],
                                         [0.1, 0.2, 0.1, 0.7, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]])
        self.label_predictions = torch.Tensor([0, 1, 1, 1, 3, 1])
        self.targets = torch.Tensor([0, 4, 1, 0, 3, 0])
        self.accuracy = (self.label_predictions == self.targets).float().mean() * 100

    def test_standard(self):
        accuracy = Accuracy()
        actual = accuracy(self.predictions, self.targets)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.accuracy, actual)

    def test_functional(self):
        actual = acc(self.predictions, self.targets)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.accuracy, actual)

    def test_hundred(self):
        accuracy = Accuracy()
        actual = accuracy(self.predictions, self.label_predictions)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(100., actual)

    def test_sum(self):
        accuracy = Accuracy(reduction='sum')
        actual = accuracy(self.predictions, self.targets)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.accuracy * len(self.predictions), actual)

    def test_functional_sum(self):
        actual = acc(self.predictions, self.targets, reduction='sum')
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.accuracy * len(self.predictions), actual)

    def test_sum_hundred(self):
        accuracy = Accuracy(reduction='sum')
        actual = accuracy(self.predictions, self.label_predictions)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(100. * len(self.predictions), actual)

    def test_none(self):
        accuracy = Accuracy(reduction='none')
        actual = accuracy(self.predictions, self.targets)
        self.assertEqual(self.targets.shape, actual.shape)
        self.assertTrue(torch.all(100. * (self.label_predictions == self.targets).float() == actual))

    def test_functional_none(self):
        actual = acc(self.predictions, self.targets, reduction='none')
        self.assertEqual(self.targets.shape, actual.shape)
        self.assertTrue(torch.all(100. * (self.label_predictions == self.targets).float() == actual))

    def test_none_hundred(self):
        accuracy = Accuracy(reduction='none')
        actual = accuracy(self.predictions, self.label_predictions)
        self.assertEqual(self.targets.shape, actual.shape)
        self.assertTrue(torch.all(100. * torch.ones_like(self.label_predictions).float() == actual))

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_on_gpu(self):
        accuracy = Accuracy()
        accuracy.cuda()
        actual = accuracy(self.predictions.cuda(), self.targets.cuda())
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.accuracy, actual.cpu())

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_functional_on_gpu(self):
        actual = acc(self.predictions.cuda(), self.targets.cuda())
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.accuracy, actual.cpu())


class IgnoreIndexAccuracyTest(TestCase):

    def setUp(self):
        self.predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0],
                                         [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.5, 0.1, 0.2, 0.0],
                                         [0.1, 0.2, 0.1, 0.7, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]])
        self.label_predictions = torch.Tensor([0, 1, 1, 1, 3, 1])
        self.targets = torch.Tensor([-100, 4, 1, -100, 3, 0])
        self.accuracy = 50.

    def test_standard(self):
        accuracy = Accuracy()
        actual = accuracy(self.predictions, self.targets)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.accuracy, actual)

    def test_functional_standard(self):
        actual = acc(self.predictions, self.targets)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.accuracy, actual)

    def test_ignore_index_with_different_value(self):
        accuracy = Accuracy(ignore_index=-1)
        actual = accuracy(self.predictions, self.targets)
        self.assertEqual((), actual.shape)
        self.assertAlmostEqual(1 / 3 * 100, float(actual), places=5)

    def test_functional_ignore_index_with_different_value(self):
        actual = acc(self.predictions, self.targets, ignore_index=-1)
        self.assertEqual((), actual.shape)
        self.assertAlmostEqual(1 / 3 * 100, float(actual), places=5)

    def test_ignore_index_with_valid_index(self):
        accuracy = Accuracy(ignore_index=1)
        actual = accuracy(self.predictions, self.targets)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(20., actual)

    def test_functional_ignore_index_with_valid_index(self):
        actual = acc(self.predictions, self.targets, ignore_index=1)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(20., actual)


class BinaryAccuracyTest(TestCase):

    def setUp(self):
        self.predictions = torch.Tensor([-0.45, -0.66, 0.8, -1.65, 0.42, -0.04, -0.99, -0.46, 1.12, 1.93])
        self.label_predictions = torch.Tensor([0., 0., 1., 0., 1., 0., 0., 0., 1., 1.])
        self.targets = torch.Tensor([1., 0., 0., 0., 1., 0., 0., 1., 1., 0.])
        self.binary_accuracy = (self.label_predictions == self.targets).float().mean() * 100

    def test_standard(self):
        binary_accuracy = BinaryAccuracy()
        actual = binary_accuracy(self.predictions, self.targets)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.binary_accuracy, actual)

    def test_functional(self):
        actual = bin_acc(self.predictions, self.targets)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.binary_accuracy, actual)

    def test_hundred(self):
        binary_accuracy = BinaryAccuracy()
        actual = binary_accuracy(self.predictions, self.label_predictions)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(100., actual)

    def test_sum(self):
        binary_accuracy = BinaryAccuracy(reduction='sum')
        actual = binary_accuracy(self.predictions, self.targets)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal((self.label_predictions == self.targets).float().sum() * 100, actual)

    def test_functional_sum(self):
        actual = bin_acc(self.predictions, self.targets, reduction='sum')
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal((self.label_predictions == self.targets).float().sum() * 100, actual)

    def test_sum_hundred(self):
        binary_accuracy = BinaryAccuracy(reduction='sum')
        actual = binary_accuracy(self.predictions, self.label_predictions)
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(100. * len(self.predictions), actual)

    def test_none(self):
        binary_accuracy = BinaryAccuracy(reduction='none')
        actual = binary_accuracy(self.predictions, self.targets)
        self.assertEqual(self.targets.shape, actual.shape)
        self.assertTrue(torch.all(100. * (self.label_predictions == self.targets).float() == actual))

    def test_functional_none(self):
        actual = bin_acc(self.predictions, self.targets, reduction='none')
        self.assertEqual(self.targets.shape, actual.shape)
        self.assertTrue(torch.all(100. * (self.label_predictions == self.targets).float() == actual))

    def test_none_hundred(self):
        binary_accuracy = BinaryAccuracy(reduction='none')
        actual = binary_accuracy(self.predictions, self.label_predictions)
        self.assertEqual(self.targets.shape, actual.shape)
        self.assertTrue(torch.all(100. * torch.ones_like(self.label_predictions).float() == actual))

    def test_threshold(self):
        threshold = -0.5
        binary_accuracy = BinaryAccuracy(threshold=threshold)
        actual = binary_accuracy(self.predictions, self.targets)
        label_predictions = (self.predictions > threshold).float()
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal((label_predictions == self.targets).float().mean() * 100, actual)

    def test_functional_threshold(self):
        threshold = -0.5
        actual = bin_acc(self.predictions, self.targets, threshold=threshold)
        label_predictions = (self.predictions > threshold).float()
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal((label_predictions == self.targets).float().mean() * 100, actual)

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_on_gpu(self):
        binary_accuracy = BinaryAccuracy()
        binary_accuracy.cuda()
        actual = binary_accuracy(self.predictions.cuda(), self.targets.cuda())
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.binary_accuracy, actual.cpu())

    @skipIf(not torch.cuda.is_available(), "no gpu available")
    def test_functional_on_gpu(self):
        actual = bin_acc(self.predictions.cuda(), self.targets.cuda())
        self.assertEqual((), actual.shape)
        np.testing.assert_almost_equal(self.binary_accuracy, actual.cpu())
