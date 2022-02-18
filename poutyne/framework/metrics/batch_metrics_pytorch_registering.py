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
