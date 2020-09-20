import torch.nn.functional as F

from .metrics_registering import register_batch_metric, register_batch_metric_function

register_batch_metric_function(F.cross_entropy)
register_batch_metric_function(F.kl_div)


@register_batch_metric('binary_cross_entropy', 'bce')
def bce(y_pred, y_true):
    return F.binary_cross_entropy(y_pred, y_true)


@register_batch_metric('binary_cross_entropy_with_logits', 'bce_with_logits')
def bce_with_logits(y_pred, y_true):
    return F.binary_cross_entropy_with_logits(y_pred, y_true)


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
