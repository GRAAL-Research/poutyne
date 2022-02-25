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

from typing import Dict

import numpy as np

from .callbacks import Callback


class TerminateOnNaN(Callback):
    """
    Stops the training when the loss is either `NaN` or `inf`.
    """

    def on_train_batch_end(self, batch_number: int, logs: Dict):
        loss = logs['loss']
        if np.isnan(loss) or np.isinf(loss):
            print(f'Batch {batch_number:d}: Invalid loss, terminating training')
            self.model.stop_training = True
