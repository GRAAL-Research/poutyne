from unittest import TestCase

import numpy
import torch

from poutyne.framework.metrics import F1

fake_predictions = torch.Tensor([[0.35, 0.25, 0.1, 0.1, 0.2], [0.1, 0.6, 0.1, 0.2, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0],
                                 [0.1, 0.5, 0.1, 0.2, 0.0], [0.1, 0.2, 0.1, 0.7, 0.0], [0.1, 0.6, 0.1, 0.2, 0.0]])
fake_targets = torch.Tensor([0, 4, 1, 0, 3, 0])


class ModelTest(TestCase):
    def test_F1Metric_micro_average_metric(self):
        metric = F1()
        metric(fake_predictions, fake_targets)
        fscores = metric.get_metric()

        true_positives = [1, 1, 0, 1, 0]
        false_positives = [0, 3, 0, 0, 0]
        false_negatives = [2, 0, 0, 0, 1]
        mean_true_positive = numpy.mean(true_positives)
        mean_false_positive = numpy.mean(false_positives)
        mean_false_negative = numpy.mean(false_negatives)

        micro_precision = mean_true_positive / (mean_true_positive + mean_false_positive)
        micro_recall = mean_true_positive / (mean_true_positive + mean_false_negative)
        micro_fscore = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

        numpy.testing.assert_almost_equal(fscores, micro_fscore, decimal=2)

    def test_F1Metric_macro_average_metric(self):
        metric = F1(average='macro')
        metric(fake_predictions, fake_targets)
        fscores = metric.get_metric()

        desired_precisions = [1.00, 0.25, 0.00, 1.00, 0.00]
        desired_recalls = [0.33, 1.00, 0.00, 1.00, 0.00]
        desired_fscores = [(2 * p * r) / (p + r) if p + r != 0.0 else 0.0
                           for p, r in zip(desired_precisions, desired_recalls)]

        macro_fscore = numpy.mean(desired_fscores)

        numpy.testing.assert_almost_equal(fscores, macro_fscore, decimal=2)
