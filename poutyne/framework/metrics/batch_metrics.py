import torch.nn as nn
import torch.nn.functional as F

from .metrics_registering import register_batch_metric, register_batch_metric_function


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

    def __init__(self, ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__(reduction)
        self.__name__ = 'acc'
        self.ignore_index = ignore_index

    def forward(self, y_pred, y_true):
        return acc(y_pred, y_true, ignore_index=self.ignore_index, reduction=self.reduction)

@register_batch_metric('acc', 'accuracy')
def acc(y_pred, y_true, ignore_index=-100, reduction='mean'):
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

    def __init__(self, threshold: float = 0., reduction: str = 'mean'):
        super().__init__(reduction)
        self.__name__ = 'bin_acc'
        self.threshold = threshold

    def forward(self, y_pred, y_true):
        return bin_acc(y_pred, y_true, threshold=self.threshold, reduction=self.reduction)


@register_batch_metric('binacc', 'binaryacc', 'binaryaccuracy')
def bin_acc(y_pred, y_true, threshold=0., reduction='mean'):
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


def bce(y_pred, y_true):
    return F.binary_cross_entropy(y_pred, y_true)


def bce_with_logits(y_pred, y_true):
    return F.binary_cross_entropy_with_logits(y_pred, y_true)


register_batch_metric_function(F.cross_entropy)
register_batch_metric_function(bce, ['binary_cross_entropy', 'bce'])
register_batch_metric_function(bce_with_logits, ['binary_cross_entropy_with_logits', 'bce_with_logits'])
register_batch_metric_function(F.kl_div)


@register_batch_metric
def poisson_nll(y_pred, y_true):
    return F.poisson_nll_loss(y_pred, y_true)


@register_batch_metric
def hinge_embedding(y_pred, y_true):
    return F.hinge_embedding_loss(y_pred, y_true)


@register_batch_metric
def l1(y_pred, y_true):
    return F.l1_loss(y_pred, y_true)


@register_batch_metric
def mse(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)


@register_batch_metric
def multilabel_margin(y_pred, y_true):
    return F.multilabel_margin_loss(y_pred, y_true)


@register_batch_metric
def multilabel_soft_margin(y_pred, y_true):
    return F.multilabel_soft_margin_loss(y_pred, y_true)


@register_batch_metric
def multi_margin(y_pred, y_true):
    return F.multi_margin_loss(y_pred, y_true)


@register_batch_metric
def nll(y_pred, y_true):
    return F.nll_loss(y_pred, y_true)


@register_batch_metric
def smooth_l1(y_pred, y_true):
    return F.smooth_l1_loss(y_pred, y_true)


@register_batch_metric
def soft_margin(y_pred, y_true):
    return F.soft_margin_loss(y_pred, y_true)
