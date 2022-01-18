import torch.nn.functional as F

from .metrics_registering import register_batch_metric_function

register_batch_metric_function(F.cross_entropy)
register_batch_metric_function(F.kl_div)
register_batch_metric_function(F.binary_cross_entropy, ['binary_cross_entropy', 'bce'], unique_name='bce')
register_batch_metric_function(
    F.binary_cross_entropy_with_logits,
    ['binary_cross_entropy_with_logits', 'bce_with_logits'],
    unique_name='bce_with_logits',
)
register_batch_metric_function(F.poisson_nll_loss, unique_name='poisson_nll')
register_batch_metric_function(F.hinge_embedding_loss, unique_name='hinge_embedding')
register_batch_metric_function(F.l1_loss, unique_name='l1')
register_batch_metric_function(F.mse_loss, unique_name='mse')
register_batch_metric_function(F.multilabel_margin_loss, unique_name='multilabel_margin')
register_batch_metric_function(F.multilabel_soft_margin_loss, unique_name='multilabel_soft_margin')
register_batch_metric_function(F.multi_margin_loss, unique_name='multi_margin')
register_batch_metric_function(F.nll_loss, unique_name='nll')
register_batch_metric_function(F.smooth_l1_loss, unique_name='smooth_l1')
register_batch_metric_function(F.soft_margin_loss, unique_name='soft_margin')
