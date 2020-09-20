# Because nn.Module has the abstract method _forward_unimplemented
# pylint: disable=abstract-method
import torch.nn as nn

from .metrics_registering import register_batch_metric


class BatchMetric(nn.Module):

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


    Possible string name in :class:`batch_metrics argument <poutyne.Model>`:
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


@register_batch_metric('acc', 'accuracy')
def acc(y_pred, y_true, *, ignore_index=-100, reduction='mean'):
    """
    Computes the accuracy.

    This is a functionnal version of :class:`~poutyne.Accuracy`.

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

    Possible string name in :class:`batch_metrics argument <poutyne.Model>`:
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

    def __init__(self, *, threshold: float = 0., reduction: str = 'mean'):
        super().__init__(reduction)
        self.__name__ = 'bin_acc'
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        return bin_acc(y_pred, y_true, threshold=self.threshold, reduction=self.reduction)


@register_batch_metric('binacc', 'binaryacc', 'binaryaccuracy')
def bin_acc(y_pred, y_true, *, threshold=0., reduction='mean'):
    """
    Computes the binary accuracy.

    This is a functionnal version of :class:`~poutyne.BinaryAccuracy`.

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


    Possible string name in :class:`batch_metrics argument <poutyne.Model>`:
        - ``'top1'``
        - ``'top1_acc'``
        - ``'top1_accuracy'``
        - ``'top5'``
        - ``'top5_acc'``
        - ``'top5_accuracy'``

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

    This is a functionnal version of :class:`~poutyne.TopKAccuracy`.

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


@register_batch_metric('top1', 'top1acc', 'top1accuracy')
def top1(y_pred, y_true, **kwargs):
    return acc(y_pred, y_true, **kwargs)


@register_batch_metric('top5', 'top5acc', 'top5accuracy')
def top5(y_pred, y_true, **kwargs):
    return topk(y_pred, y_true, 5, **kwargs)
