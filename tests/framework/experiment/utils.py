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

from poutyne import Callback


class ConstantMetric:
    __name__ = 'const'

    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, *args, **kwds):
        return self.value


class ConstantMetricCallback(Callback):
    def __init__(self, values, constant_metric):
        super().__init__()
        self.values = values
        self.constant_metric = constant_metric

    def on_epoch_begin(self, epoch_number, logs):
        self.constant_metric.value = (
            self.values[epoch_number - 1] if epoch_number < len(self.values) else self.values[-1]
        )
