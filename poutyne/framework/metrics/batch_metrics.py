import torch.nn.functional as F

from .metrics_registering import register_batch_metric, register_batch_metric_function


@register_batch_metric('acc', 'accuracy')
def acc(y_pred, y_true, ignore_index=-100):
    y_pred = y_pred.argmax(1)
    weights = (y_true != ignore_index).float()
    num_labels = weights.sum()
    acc_pred = ((y_pred == y_true).float() * weights).sum() / num_labels
    return acc_pred * 100


@register_batch_metric('binacc', 'binaryacc', 'binaryaccuracy')
def bin_acc(y_pred, y_true, threshold=0.):
    y_pred = (y_pred > threshold).float()
    acc_pred = (y_pred == y_true).float().mean()
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
