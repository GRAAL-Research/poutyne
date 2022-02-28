"""
The source code of this file was copied from the AllenNLP project, and has been modified. All modifications
made from the original source code are under the LGPLv3 license.


Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information on the Poutyne and AllenNLP repository.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.


Copyright 2019 AllenNLP

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import warnings
from typing import Optional, Union, List, Tuple
import torch
from ..base import Metric
from ..metrics_registering import register_metric_class


class FBeta(Metric):
    """
    The source code of this class is under the Apache v2 License and was copied from
    the AllenNLP project and has been modified.

    Compute precision, recall, F-measure and support for each class.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    If we have precision and recall, the F-beta score is simply:
    ``F-beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)``

    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.

    The support is the number of occurrences of each class in ``y_true``.

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'{metric}_{average}'``
        - Validation: ``'val_{metric}_{average}'``

        where ``{metric}`` and ``{average}`` are replaced by the value of their
        respective parameters.

    Args:
        metric (Optional[str]): One of {'fscore', 'precision', 'recall'}.
            Whether to return the F-score, the precision or the recall. When not
            provided, all three metrics are returned. (Default value = None)
        average (Union[str, int]): One of {'micro' (default), 'macro', label_number}
            If the argument is of type integer, the score for this class (the label number) is calculated.
            Otherwise, this determines the type of averaging performed on the data:

            ``'binary'``:
                Calculate metrics with regard to a single class identified by the
                `pos_label` argument. This is equivalent to `average=pos_label` except
                that the binary mode is enforced, i.e. an exception will be raised if
                there are more than two prediction scores.
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.

            (Default value = 'macro')
        beta (float):
            The strength of recall versus precision in the F-score. (Default value = 1.0)
        pos_label (int):
            The class with respect to which the metric is computed when ``average == 'binary'``. Otherwise, this
            argument has no effect. (Default value = 1)
        ignore_index (int): Specifies a target value that is ignored. This also works in combination with
            a mask if provided. (Default value = -100)
        threshold (float): Threshold for when there is a single score for each prediction. If a sigmoid output is used,
            this should be between 0 and 1. A suggested value would be 0.5. If a logits output is used, the threshold
            would be between -inf and inf. The suggested default value is 0 as to give a probability of 0.5 if a sigmoid
            output were used. (Default = 0)
        names (Optional[Union[str, List[str]]]): The names associated to the metrics. It is a string when
            a single metric is requested. It is a list of 3 strings if all metrics are requested.
            (Default value = None)
    """

    def __init__(
        self,
        *,
        metric: Optional[str] = None,
        average: Union[str, int] = 'macro',
        beta: float = 1.0,
        pos_label: int = 1,
        ignore_index: int = -100,
        threshold: float = 0.0,
        names: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__()
        self.metric_options = ('fscore', 'precision', 'recall')
        if metric is not None and metric not in self.metric_options:
            raise ValueError(f"`metric` has to be one of {self.metric_options}.")

        if metric in ('precision', 'recall') and beta != 1.0:
            warnings.warn(f"The use of the `beta` argument is useless with {repr(metric)}.")

        average_options = ('binary', 'micro', 'macro')
        if average not in average_options and not isinstance(average, int):
            raise ValueError(f"`average` has to be one of {average_options} or an integer.")

        if beta <= 0:
            raise ValueError("`beta` should be >0 in the F-beta score.")

        self._metric = metric
        self._average = average if average in average_options else None
        self._label = None
        if isinstance(average, int):
            self._label = average
        elif average == 'binary':
            self._label = pos_label
        self._beta = beta
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.__name__ = self._get_names(names)

        # statistics
        # the total number of true positive instances under each class
        # Shape: (num_classes, )
        self.register_buffer('_true_positive_sum', None)
        # the total number of instances
        # Shape: (num_classes, )
        self.register_buffer('_total_sum', None)
        # the total number of instances under each _predicted_ class,
        # including true positives and false positives
        # Shape: (num_classes, )
        self.register_buffer('_pred_sum', None)
        # the total number of instances under each _true_ class,
        # including true positives and false negatives
        # Shape: (num_classes, )
        self.register_buffer('_true_sum', None)

    def _get_name(self, metric):
        name = metric
        if self._average is not None:
            name += '_' + self._average
        if self._label is not None:
            name += '_' + str(self._label)
        return name

    def _get_names(self, names):
        if self._metric is None:
            default_name = list(map(self._get_name, self.metric_options))
        else:
            default_name = self._get_name(self._metric)

        if names is not None:
            self._validate_supplied_names(names, default_name)
            return names

        return default_name

    def _validate_supplied_names(self, names, default_name):
        names_list = [names] if isinstance(names, str) else names
        default_name = [default_name] if isinstance(default_name, str) else default_name
        if len(names_list) != len(default_name):
            raise ValueError(f"`names` should contain names for the following metrics: {', '.join(default_name)}.")

    def forward(
        self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[float, Tuple[float]]:
        """
        Update the confusion matrix for calculating the F-score and compute the metrics for the current batch. See
        :meth:`FBeta.compute` for details on the return value.

        Args:
            y_pred (torch.Tensor): A tensor of predictions of shape (batch_size, num_classes, ...).
            y_true (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                Ground truths. A tensor of the integer class label of shape (batch_size, ...). It must
                be the same shape as the ``y_pred`` tensor without the ``num_classes`` dimension.
                It can also be a tuple with two tensors of the same shape, the first being the
                ground truths and the second being a mask.

        Returns:
            A float if a single metric is set in the ``__init__`` or a tuple of floats (f-score, precision, recall) if
            all metrics are requested.
        """
        true_positive_sum, pred_sum, true_sum = self._update(y_pred, y_true)
        return self._compute(true_positive_sum, pred_sum, true_sum)

    def update(self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """
        Update the confusion matrix for calculating the F-score.

        Args:
            y_pred (torch.Tensor): A tensor of predictions of shape (batch_size, num_classes, ...).
            y_true (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                Ground truths. A tensor of the integer class label of shape (batch_size, ...). It must
                be the same shape as the ``y_pred`` tensor without the ``num_classes`` dimension.
                It can also be a tuple with two tensors of the same shape, the first being the
                ground truths and the second being a mask.
        """
        self._update(y_pred, y_true)

    def _update(self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        # pylint: disable=too-many-branches
        if isinstance(y_true, tuple):
            y_true, mask = y_true
            mask = mask.bool()
        else:
            mask = torch.ones_like(y_true).bool()

        if self.ignore_index is not None:
            mask *= y_true != self.ignore_index

        if y_pred.shape[0] == 1:
            y_pred, y_true, mask = (
                y_pred.squeeze().unsqueeze(0),
                y_true.squeeze().unsqueeze(0),
                mask.squeeze().unsqueeze(0),
            )
        else:
            y_pred, y_true, mask = y_pred.squeeze(), y_true.squeeze(), mask.squeeze()

        num_classes = 2
        if y_pred.shape != y_true.shape:
            num_classes = y_pred.size(1)

        if (y_true >= num_classes).any():
            raise ValueError(
                f"A gold label passed to FBetaMeasure contains an id >= {num_classes}, the number of classes."
            )

        if self._average == 'binary' and num_classes > 2:
            raise ValueError("When `average` is binary, the number of prediction scores must be 2.")

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes, device=y_pred.device)
            self._true_sum = torch.zeros(num_classes, device=y_pred.device)
            self._pred_sum = torch.zeros(num_classes, device=y_pred.device)
            self._total_sum = torch.zeros(num_classes, device=y_pred.device)

        y_true = y_true.float()

        if y_pred.shape != y_true.shape:
            argmax_y_pred = y_pred.argmax(1).float()
        else:
            argmax_y_pred = (y_pred > self.threshold).float()
        true_positives = (y_true == argmax_y_pred) * mask
        true_positives_bins = y_true[true_positives]

        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes, device=y_pred.device)
        else:
            true_positive_sum = torch.bincount(true_positives_bins.long(), minlength=num_classes).float()

        pred_bins = argmax_y_pred[mask].long()
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes, device=y_pred.device)

        y_true_bins = y_true[mask].long()
        if y_true.shape[0] != 0:
            true_sum = torch.bincount(y_true_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes, device=y_pred.device)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum
        self._total_sum += mask.sum().to(torch.float)

        return true_positive_sum, pred_sum, true_sum

    def compute(self) -> Union[float, Tuple[float]]:
        """
        Returns either a float if a single metric is set in the ``__init__`` or a tuple
        of floats (f-score, precision, recall) if all metrics are requested.
        """
        if self._true_positive_sum is None:
            raise RuntimeError("You never call this metric before.")

        return self._compute(self._true_positive_sum, self._pred_sum, self._true_sum)

    def _compute(self, tp_sum, pred_sum, true_sum):
        if self._average == 'micro':
            tp_sum = tp_sum.sum()
            pred_sum = pred_sum.sum()
            true_sum = true_sum.sum()

        beta2 = self._beta**2
        # Finally, we have all our sufficient statistics.
        precision = _prf_divide(tp_sum, pred_sum)
        recall = _prf_divide(tp_sum, true_sum)
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[tp_sum == 0] = 0.0

        if self._average == 'macro':
            precision = precision.mean()
            recall = recall.mean()
            fscore = fscore.mean()

        if self._label is not None:
            # Retain only selected labels and order them
            precision = precision[self._label]
            recall = recall[self._label]
            fscore = fscore[self._label]

        if self._metric is None:
            return [fscore.item(), precision.item(), recall.item()]

        if self._metric == 'fscore':
            return fscore.item()
        if self._metric == 'precision':
            return precision.item()
        # if self._metric == 'recall':
        return recall.item()

    def reset(self) -> None:
        self._true_positive_sum = None
        self._pred_sum = None
        self._true_sum = None
        self._total_sum = None


@register_metric_class
class F1(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'fscore'`` and ``beta == 1``.

    Possible string name:
        - ``'f1'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'fscore_{average}'``
        - Validation: ``'val_fscore_{average}'``

        where ``{average}`` is replaced by the value of the respective parameter.
    """

    def __init__(self, **kwargs):
        _raise_invalid_use_of_beta('F1', kwargs)
        super().__init__(metric='fscore', **kwargs)


@register_metric_class
class Precision(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'precision'``.

    Possible string name:
        - ``'precision'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'precision_{average}'``
        - Validation: ``'val_precision_{average}'``

        where ``{average}`` is replaced by the value of the respective parameter.
    """

    def __init__(self, **kwargs):
        super().__init__(metric='precision', **kwargs)


@register_metric_class
class Recall(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'recall'``.

    Possible string name:
        - ``'recall'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'recall_{average}'``
        - Validation: ``'val_recall_{average}'``

        where ``{average}`` is replaced by the value of the respective parameter.
    """

    def __init__(self, **kwargs):
        super().__init__(metric='recall', **kwargs)


@register_metric_class('binaryf1', 'binf1')
class BinaryF1(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'fscore'``, ``average='binary'`` and ``beta == 1``.

    Possible string name:
        - ``'binary_f1'``
        - ``'bin_f1'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'bin_fscore'``
        - Validation: ``'val_bin_fscore'``
    """

    def __init__(self, **kwargs):
        _raise_invalid_use_of_beta('BinaryF1', kwargs)
        kwargs = {'names': 'bin_fscore', **kwargs}
        super().__init__(metric='fscore', average='binary', **kwargs)


@register_metric_class('binaryprecision', 'binprecision')
class BinaryPrecision(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'precision'`` and ``average='binary'``.

    Possible string name:
        - ``'binary_precision'``
        - ``'bin_precision'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'bin_precision'``
        - Validation: ``'val_bin_precision'``
    """

    def __init__(self, **kwargs):
        kwargs = {'names': 'bin_precision', **kwargs}
        super().__init__(metric='precision', average='binary', **kwargs)


@register_metric_class('binaryrecall', 'binrecall')
class BinaryRecall(FBeta):
    """
    Alias class for :class:`~poutyne.FBeta` where ``metric == 'recall'`` and ``average='binary'``.

    Possible string name:
        - ``'binary_recall'``
        - ``'bin_recall'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'bin_recall'``
        - Validation: ``'val_bin_recall'``
    """

    def __init__(self, **kwargs):
        kwargs = {'names': 'bin_recall', **kwargs}
        super().__init__(metric='recall', average='binary', **kwargs)


def _prf_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result


def _raise_invalid_use_of_beta(name, kwargs):
    if 'beta' in kwargs:
        raise ValueError(f"The use of the `beta` argument with {name} is invalid. Use FBeta instead.")
