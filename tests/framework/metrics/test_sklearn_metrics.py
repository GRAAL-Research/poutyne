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

    def _get_data_loader(self, *tensors):
        return DataLoader(TensorDataset(*tensors), batch_size=SKLearnMetricsTest.batch_size)

    def test_classification(self):
        self._test_classification(roc_auc_score, True)
        self._test_classification(roc_auc_score, False)

        self._test_classification(average_precision_score, True)
        self._test_classification(average_precision_score, False)

        two_skl_metrics = [roc_auc_score, average_precision_score]
        self._test_classification(two_skl_metrics, True)
        self._test_classification(two_skl_metrics, False)

    def test_multiclass_classification_with_kwargs(self):
        roc_auc_kwargs = dict(multi_class='ovr', average='macro')
        self._test_multiclass_classification(roc_auc_score, True, kwargs=roc_auc_kwargs)
        self._test_multiclass_classification(roc_auc_score, False, kwargs=roc_auc_kwargs)

    def test_classification_with_custom_names(self):
        roc_names = 'roc'
        self._test_classification(roc_auc_score, True, names=roc_names)
        self._test_classification(roc_auc_score, False, names=roc_names)

        ap_names = 'ap'
        self._test_classification(average_precision_score, True, names=ap_names)
        self._test_classification(average_precision_score, False, names=ap_names)

        two_names = ['roc', 'ap']
        two_skl_metrics = [roc_auc_score, average_precision_score]
        self._test_classification(two_skl_metrics, True, names=two_names)
        self._test_classification(two_skl_metrics, False, names=two_names)

    def test_regression(self):
        self._test_regression(r2_score, True)
        self._test_regression(r2_score, False)

        self._test_regression(gini, True)
        self._test_regression(gini, False)

        two_skl_metrics = [r2_score, gini]
        self._test_regression(two_skl_metrics, True)
        self._test_regression(two_skl_metrics, False)

    def _test_classification(self, sklearn_metrics, with_sample_weight, *, kwargs=None, names=None):
        return self._test_epoch_metric(self.classification_pred,
                                       self.classification_true,
                                       sklearn_metrics,
                                       with_sample_weight,
                                       kwargs=kwargs,
                                       names=names)

    def _test_multiclass_classification(self, sklearn_metrics, with_sample_weight, *, kwargs=None, names=None):
        return self._test_epoch_metric(self.multiclass_classification_pred,
                                       self.multiclass_classification_true,
                                       sklearn_metrics,
                                       with_sample_weight,
                                       kwargs=kwargs,
                                       names=names)

    def _test_regression(self, sklearn_metrics, with_sample_weight, *, kwargs=None, names=None):
        return self._test_epoch_metric(self.regression_pred,
                                       self.regression_true,
                                       sklearn_metrics,
                                       with_sample_weight,
                                       kwargs=kwargs,
                                       names=names)

    def _test_epoch_metric(self, pred, true, sklearn_metrics, with_sample_weight, *, kwargs=None, names=None):
        epoch_metric = SKLearnMetrics(sklearn_metrics, kwargs=kwargs, names=names)

        if with_sample_weight:
            loader = self._get_data_loader(pred, (true, self.sample_weight))
            sample_weight = self.sample_weight.numpy()
        else:
            loader = self._get_data_loader(pred, true)
            sample_weight = None

        true = true.numpy()
        pred = pred.numpy()

        if not isinstance(sklearn_metrics, list):
            sklearn_metrics = [sklearn_metrics]

        if kwargs is not None and not isinstance(kwargs, list):
            kwargs = [kwargs]
        kwargs = kwargs if kwargs is not None else repeat({})

        if names is not None and not isinstance(names, list):
            names = [names]
        names = [func.__name__ for func in sklearn_metrics] if names is None else names

        expected = {
            name: f(true, pred, sample_weight=sample_weight, **kw)
            for name, f, kw in zip(names, sklearn_metrics, kwargs)
        }

        with torch.no_grad():
            for y_pred, y_true in loader:
                epoch_metric(y_pred, y_true)
        actual = epoch_metric.get_metric()
        self.assertEqual(expected, actual)
