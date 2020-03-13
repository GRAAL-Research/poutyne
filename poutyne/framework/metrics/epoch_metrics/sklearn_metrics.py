from typing import Optional, Union, List, Callable
from itertools import repeat
import numpy as np
from .base import EpochMetric


class SKLearnMetrics(EpochMetric):
    """
    Wrap metrics with scikit-learn-like interface
    (``metric(y_true, y_pred, sample_weight=sample_weight, **kwargs)``).
    The ``SKLearnMetrics`` object has to keep in memory the ground truths and
    predictions so that in can compute the metric at the end.

    Example:
        .. code-block:: python

            from sklearn.metrics import roc_auc_score, average_precision_score
            from poutyne.framework.metrics import SKLearnMetrics
            my_epoch_metric = SKLearnMetrics([roc_auc_score, average_precision_score])

    Args:
        funcs (Union[Callable, List[Callable]]): A metric or a list of metrics with a
            scikit-learn-like interface.
        kwargs (Optional[Union[dict, List[dict]]]): Optional list containing dictionaries corresponding to
            keyword arguments to pass to each corresponding function. (Default value = None)
    """

    def __init__(self, funcs: Union[Callable, List[Callable]],
                 kwargs: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__()
        self.funcs = funcs if isinstance(funcs, (list, tuple)) else [funcs]
        self.kwargs = None
        if kwargs is not None:
            self.kwargs = kwargs if isinstance(kwargs, (list, tuple)) else [kwargs]
        self.__name__ = [func.__name__ for func in self.funcs]

        if self.kwargs is not None and len(self.funcs) != len(self.kwargs):
            raise ValueError("`kwargs` has to have the same length as `funcs` when provided")

        self.reset()

    def forward(self, y_pred, y_true):
        self.y_pred_list.append(y_pred.cpu().numpy())
        if isinstance(y_true, (tuple, list)):
            y_true, sample_weight = y_true
            self.sample_weight_list.append(sample_weight.cpu().numpy())
        self.y_true_list.append(y_true.cpu().numpy())

    def get_metric(self):
        sample_weight = None
        if len(self.sample_weight_list) != 0:
            sample_weight = np.concatenate(self.sample_weight_list)
        y_pred = np.concatenate(self.y_pred_list)
        y_true = np.concatenate(self.y_true_list)

        self.reset()

        kwargs = self.kwargs if self.kwargs is not None else repeat({})
        return {
            func.__name__: func(y_true, y_pred, sample_weight=sample_weight, **kwargs)
            for func, kwargs in zip(self.funcs, kwargs)
        }

    def reset(self) -> None:
        self.y_true_list = []
        self.y_pred_list = []
        self.sample_weight_list = []
