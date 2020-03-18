from typing import Optional, Union, List, Callable, Dict
import numpy as np
from .base import EpochMetric


class SKLearnMetrics(EpochMetric):
    """
    Wrap metrics with Scikit-learn-like interface
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
        kwargs (Optional[Union[dict, List[dict]]]): Optional dictionary of list of dictionaries
            corresponding to keyword arguments to pass to each corresponding metric.
            (Default value = None)
        names (Optional[Union[str, List[str]]]): Optional string or list of strings corresponding to
            the names given to the metrics. (Default value = None)
    """

    def __init__(self,
                 funcs: Union[Callable, List[Callable]],
                 kwargs: Optional[Union[dict, List[dict]]] = None,
                 names: Optional[Union[str, List[str]]] = None) -> None:
        super().__init__()

        funcs = funcs if isinstance(funcs, (list, tuple)) else [funcs]

        if kwargs is not None:
            kwargs = kwargs if isinstance(kwargs, (list, tuple)) else [kwargs]
            if kwargs is not None and len(funcs) != len(kwargs):
                raise ValueError("`kwargs` has to have the same length as `funcs` when provided")

        if names is not None:
            names = names if isinstance(names, (list, tuple)) else [names]
            if len(funcs) != len(names):
                raise ValueError("`names` has to have the same length as `funcs` when provided")

        self.funcs = funcs
        self.kwargs = [{}] * len(self.funcs) if kwargs is None else kwargs
        self.__name__ = [func.__name__ for func in self.funcs] if names is None else names

        self.reset()

    def forward(self, y_pred, y_true):
        """
        Accumulate the predictions, ground truths and sample weights if any.

        Args:
            y_pred (torch.Tensor): A tensor of predictions of the shape expected by
                the metric functions passed to the class.
            y_true (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                Ground truths. A tensor of ground truths of the shape expected by
                the metric functions passed to the class.
                It can also be a tuple with two tensors, the first being the
                ground truths and the second corresponding the ``sample_weight``
                argument passed to the metric functions in scikit-learn.
        """
        self.y_pred_list.append(y_pred.cpu().numpy())
        if isinstance(y_true, (tuple, list)):
            y_true, sample_weight = y_true
            self.sample_weight_list.append(sample_weight.cpu().numpy())
        self.y_true_list.append(y_true.cpu().numpy())

    def get_metric(self) -> Dict:
        """
        Returns the metrics as a dictionary with the names as keys.
        """
        sample_weight = None
        if len(self.sample_weight_list) != 0:
            sample_weight = np.concatenate(self.sample_weight_list)
        y_pred = np.concatenate(self.y_pred_list)
        y_true = np.concatenate(self.y_true_list)

        self.reset()

        return {
            name: func(y_true, y_pred, sample_weight=sample_weight, **kwargs)
            for name, func, kwargs in zip(self.__name__, self.funcs, self.kwargs)
        }

    def reset(self) -> None:
        self.y_true_list = []
        self.y_pred_list = []
        self.sample_weight_list = []
