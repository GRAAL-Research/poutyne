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

import torch.optim as optim

all_optimizers_dict = dict(
    adadelta=optim.Adadelta,
    adagrad=optim.Adagrad,
    adam=optim.Adam,
    adamw=optim.AdamW,
    sparseadam=optim.SparseAdam,
    adamax=optim.Adamax,
    asgd=optim.ASGD,
    lbfgs=optim.LBFGS,
    nadam=optim.NAdam,
    radam=optim.RAdam,
    rmsprop=optim.RMSprop,
    rprop=optim.Rprop,
    sgd=optim.SGD,
)


def get_optimizer(optimizer, module):
    if isinstance(optimizer, (str, dict)):
        kwargs = {}
        if isinstance(optimizer, dict):
            optimizer = dict(optimizer)
            kwargs = optimizer
            optimizer = optimizer.pop('optim')

        optimizer = optimizer.lower()

        if optimizer == 'sgd':
            kwargs.setdefault('lr', 1e-2)

        params = (p for p in module.parameters() if p.requires_grad)
        return all_optimizers_dict[optimizer](params, **kwargs)

    return optimizer
