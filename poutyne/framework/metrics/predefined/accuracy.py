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

from abc import ABC
import torch.nn as nn

from ..metrics_registering import register_metric_func, do_register_metric_func


class BatchMetric(ABC, nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        REDUCTIONS = ['none', 'mean', 'sum']
        if reduction not in REDUCTIONS:
            raise ValueError(f"Reduction is not in {REDUCTIONS}")

        self.reduction = reduction


class Accuracy(BatchMetric):
    r"""
    This metric computes the accuracy using a similar interface to
    :class:`~torch.nn.CrossEntropyLoss`.

    Args:
        ignore_index (int): Specifies a target value that is ignored and does not contribute
            to the accuracy. (Default value = -100)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.


    Possible string name:
        - ``'acc'``
        - ``'accuracy'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'acc'``
        - Validation: ``'val_acc'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          `K`-dimensional accuracy.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional accuracy.
        - Output: The accuracy.
    """

    def __init__(self, *, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__(reduction)
        self.__name__ = 'acc'
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        return acc(y_pred, y_true, ignore_index=self.ignore_index, reduction=self.reduction)


@register_metric_func('acc', 'accuracy')
def acc(y_pred, y_true, *, ignore_index=-100, reduction='mean'):
    """
    Computes the accuracy.

    This is a functional version of :class:`~poutyne.Accuracy`.

    See :class:`~poutyne.Accuracy` for details.
    """
    y_pred = y_pred.argmax(1)
    weights = (y_true != ignore_index).float()
    num_labels = weights.sum()
    acc_pred = (y_pred == y_true).float() * weights

    if reduction in ['mean', 'sum']:
        acc_pred = acc_pred.sum()

    if reduction == 'mean':
        acc_pred = acc_pred / num_labels

    return acc_pred * 100


class BinaryAccuracy(BatchMetric):
    r"""
    This metric computes the accuracy using a similar interface to
    :class:`~torch.nn.BCEWithLogitsLoss`.

    Args:
        threshold (float): the threshold for class :math:`1`. Default value is ``0.``, that is a
            probability of ``sigmoid(0.) = 0.5``.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Possible string name:
        - ``'bin_acc'``
        - ``'binary_acc'``
        - ``'binary_accuracy'``

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'bin_acc'``
        - Validation: ``'val_bin_acc'``

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: The binary accuracy.
    """

    def __init__(self, *, threshold: float = 0.0, reduction: str = 'mean'):
        super().__init__(reduction)
        self.__name__ = 'bin_acc'
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        return bin_acc(y_pred, y_true, threshold=self.threshold, reduction=self.reduction)


@register_metric_func('binacc', 'binaryacc', 'binaryaccuracy')
def bin_acc(y_pred, y_true, *, threshold=0.0, reduction='mean'):
    """
    Computes the binary accuracy.

    This is a functional version of :class:`~poutyne.BinaryAccuracy`.

    See :class:`~poutyne.BinaryAccuracy` for details.
    """
    y_pred = (y_pred > threshold).float()
    acc_pred = (y_pred == y_true).float()
    if reduction == 'mean':
        acc_pred = acc_pred.mean()
    elif reduction == 'sum':
        acc_pred = acc_pred.sum()
    return acc_pred * 100


class TopKAccuracy(BatchMetric):
    r"""
    This metric computes the top-k accuracy using a similar interface to
    :class:`~torch.nn.CrossEntropyLoss`.

    Args:
        k (int): Specifies the value of ``k`` in the top-k accuracy.
        ignore_index (int): Specifies a target value that is ignored and does not contribute
            to the top-k accuracy. (Default value = -100)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.


    Possible string name:
        - ``'top{k}'``
        - ``'top{k}_acc'``
        - ``'top{k}_accuracy'``

        for ``{k}`` from 1 to 10, 20, 30, ..., 100.

    Keys in :class:`logs<poutyne.Callback>` dictionary of callbacks:
        - Train: ``'top{k}'``
        - Validation: ``'val_top{k}'``

        where ``{k}`` is replaced by the value of parameter ``k``.

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          `K`-dimensional top-k accuracy.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional top-k accuracy.
        - Output: The top-k accuracy.
    """

    def __init__(self, k: int, *, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__(reduction)
        self.__name__ = f'top{k}'
        self.k = k
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        return topk(y_pred, y_true, self.k, ignore_index=self.ignore_index, reduction=self.reduction)


def topk(y_pred, y_true, k, *, ignore_index=-100, reduction='mean'):
    """
    Computes the top-k accuracy.

    This is a functional version of :class:`~poutyne.TopKAccuracy`.

    See :class:`~poutyne.TopKAccuracy` for details.
    """
    topk_pred = y_pred.topk(k, dim=1)[1]
    weights = (y_true != ignore_index).float()
    num_labels = weights.sum()
    topk_acc = (y_true.unsqueeze(1) == topk_pred).any(1).float() * weights

    if reduction in ['mean', 'sum']:
        topk_acc = topk_acc.sum()

    if reduction == 'mean':
        topk_acc = topk_acc / num_labels

    return topk_acc * 100


@register_metric_func('top1', 'top1acc', 'top1accuracy')
def top1(y_pred, y_true, **kwargs):
    return acc(y_pred, y_true, **kwargs)


for k_value in range(2, 11):
    do_register_metric_func(TopKAccuracy(k_value), [f'top{k_value}', f'top{k_value}acc', f'top{k_value}accuracy'])
del k_value

for k_value in range(20, 110, 10):
    do_register_metric_func(TopKAccuracy(k_value), [f'top{k_value}', f'top{k_value}acc', f'top{k_value}accuracy'])
del k_value
