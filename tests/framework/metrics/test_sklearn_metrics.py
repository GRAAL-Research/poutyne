from unittest import TestCase, skipIf
from itertools import repeat

import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
    is_sklearn_available = True
except ImportError:
    is_sklearn_available = False
from poutyne.framework.metrics import SKLearnMetrics
from poutyne import TensorDataset


def gini(y_true, y_pred, sample_weight=None):
    sort_indices = np.argsort(y_pred)
    if sample_weight is not None:
        sample_weight = sample_weight[sort_indices]
    else:
        sample_weight = np.ones_like(y_true)
    y_true = y_true[sort_indices]
    random = (sample_weight / sample_weight.sum()).cumsum()
    lorentz = y_true.cumsum() / y_true.sum()
    return 1 - ((random[1:] - random[:-1]) * (lorentz[1:] + lorentz[:-1])).sum()


@skipIf(not is_sklearn_available, "Scikit-learn is not available")
class SKLearnMetricsTest(TestCase):
    threshold = 0.7
    steps_per_epoch = 5
    batch_size = 20
    num_examples = batch_size * steps_per_epoch

    def setUp(self):
        pred_noise = 0.1 * torch.randn((SKLearnMetricsTest.num_examples, 1))

        self.regression_pred = torch.randn((SKLearnMetricsTest.num_examples, 1))
        self.regression_true = self.regression_pred + pred_noise
        self.classification_pred = self.regression_pred
        self.classification_true = self.regression_true.sigmoid() > SKLearnMetricsTest.threshold
        self.multiclass_classification_pred = torch.randn((SKLearnMetricsTest.num_examples, 3)).softmax(1)
        self.multiclass_classification_true = self.multiclass_classification_pred.argmax(1)
        self.multiclass_errors_indices = torch.where(torch.rand(SKLearnMetricsTest.num_examples) > 0.9)[0]
        self.multiclass_classification_true[self.multiclass_errors_indices] = torch.randint(
            3, (len(self.multiclass_errors_indices), ))
        self.sample_weight = torch.rand((SKLearnMetricsTest.num_examples, 1))

        self.np_regression_pred = self.regression_pred.numpy()
        self.np_regression_true = self.regression_true.numpy()
        self.np_classification_pred = self.classification_pred.numpy()
        self.np_classification_true = self.classification_true.numpy()
        self.np_multiclass_classification_pred = self.multiclass_classification_pred.numpy()
        self.np_multiclass_classification_true = self.multiclass_classification_true.numpy()
        self.np_sample_weight = self.sample_weight.numpy()

    def _get_data_loader(self, *tensors):
        return DataLoader(TensorDataset(*tensors), batch_size=SKLearnMetricsTest.batch_size)

    def test_classification(self):
        roc_auc_epoch_metric = SKLearnMetrics(roc_auc_score)
        self._test_classification(roc_auc_epoch_metric, roc_auc_score, None, True)
        self._test_classification(roc_auc_epoch_metric, roc_auc_score, None, False)

        average_precision_epoch_metric = SKLearnMetrics(average_precision_score)
        self._test_classification(average_precision_epoch_metric, average_precision_score, None, True)
        self._test_classification(average_precision_epoch_metric, average_precision_score, None, False)

        two_skl_metrics = [roc_auc_score, average_precision_score]
        two_metrics = SKLearnMetrics(two_skl_metrics)
        self._test_classification(two_metrics, two_skl_metrics, None, True)
        self._test_classification(two_metrics, two_skl_metrics, None, False)

    def test_multiclass_classification(self):
        roc_auc_kwargs = dict(multi_class='ovr', average='macro')
        roc_auc_epoch_metric = SKLearnMetrics(roc_auc_score, kwargs=roc_auc_kwargs)
        self._test_multiclass_classification(roc_auc_epoch_metric, roc_auc_score, roc_auc_kwargs, True)
        self._test_multiclass_classification(roc_auc_epoch_metric, roc_auc_score, roc_auc_kwargs, False)

    def test_regression(self):
        r2_epoch_metric = SKLearnMetrics(r2_score)
        self._test_regression(r2_epoch_metric, r2_score, None, True)
        self._test_regression(r2_epoch_metric, r2_score, None, False)

        gini_epoch_metric = SKLearnMetrics(gini)
        self._test_regression(gini_epoch_metric, gini, None, True)
        self._test_regression(gini_epoch_metric, gini, None, False)

        two_skl_metrics = [r2_score, gini]
        two_metrics = SKLearnMetrics(two_skl_metrics)
        self._test_regression(two_metrics, two_skl_metrics, None, True)
        self._test_regression(two_metrics, two_skl_metrics, None, False)

    def _test_classification(self, epoch_metric, sklearn_metrics, kwargs, with_sample_weight):
        return self._test_epoch_metric(self.classification_pred, self.classification_true, self.np_classification_pred,
                                       self.np_classification_true, epoch_metric, sklearn_metrics, kwargs,
                                       with_sample_weight)

    def _test_multiclass_classification(self, epoch_metric, sklearn_metrics, kwargs, with_sample_weight):
        return self._test_epoch_metric(self.multiclass_classification_pred, self.multiclass_classification_true,
                                       self.np_multiclass_classification_pred, self.np_multiclass_classification_true,
                                       epoch_metric, sklearn_metrics, kwargs, with_sample_weight)

    def _test_regression(self, epoch_metric, sklearn_metrics, kwargs, with_sample_weight):
        return self._test_epoch_metric(self.regression_pred, self.regression_true, self.np_regression_pred,
                                       self.np_regression_true, epoch_metric, sklearn_metrics, kwargs,
                                       with_sample_weight)

    def _test_epoch_metric(self, pred, true, np_pred, np_true, epoch_metric, sklearn_metrics, kwargs,
                           with_sample_weight):
        # pylint: disable=too-many-arguments
        if with_sample_weight:
            loader = self._get_data_loader(pred, (true, self.sample_weight))
            np_sample_weight = self.np_sample_weight
        else:
            loader = self._get_data_loader(pred, true)
            np_sample_weight = None

        if not isinstance(sklearn_metrics, list):
            sklearn_metrics = [sklearn_metrics]
        if kwargs is not None and not isinstance(kwargs, list):
            kwargs = [kwargs]
        kwargs = kwargs if kwargs is not None else repeat({})
        expected = {
            f.__name__: f(np_true, np_pred, sample_weight=np_sample_weight, **kw)
            for f, kw in zip(sklearn_metrics, kwargs)
        }

        with torch.no_grad():
            for y_pred, y_true in loader:
                epoch_metric(y_pred, y_true)
        actual = epoch_metric.get_metric()
        self.assertEqual(expected, actual)
