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

import os
from typing import Dict, List, Union, Optional, TextIO
import csv

import torch

from ._utils import atomic_lambda_save
from .callbacks import Callback


class GradientLoggerBase(Callback):
    def __init__(self, keep_bias: bool = False, norm_type: Union[float, List[float]] = 2.0) -> None:
        super().__init__()
        self.keep_bias = keep_bias
        self.norm_type = [norm_type] if isinstance(norm_type, (float, int)) else norm_type

        self.stats_names = ['mean', 'var', 'min', 'abs_min', 'max', 'abs_max']
        self.stats_names += [f'l{norm}' for norm in self.norm_type]

        self.layers = []

    def on_train_begin(self, logs: Dict):
        self.layers = [n for n, p in self.model.network.named_parameters() if self._keep_layer(p, n)]

    def on_epoch_begin(self, epoch_number: int, logs: Dict):
        self.epoch = epoch_number

    def on_train_batch_end(self, batch_number: int, logs: Dict):
        # Just in case we want to support second-order derivatives
        with torch.no_grad():
            layer_stats = {}
            for name, param in self.model.network.named_parameters():
                if self._keep_layer(param, name):
                    grad = param.grad
                    grad_abs_values = grad.abs()

                    stats = {}
                    stats['mean'] = grad_abs_values.mean().item()
                    stats['var'] = grad_abs_values.var().item()
                    stats['min'] = grad.min().item()
                    stats['abs_min'] = grad_abs_values.min().item()
                    stats['max'] = grad.max().item()
                    stats['abs_max'] = grad_abs_values.max().item()
                    for norm in self.norm_type:
                        stats[f'l{norm}'] = grad_abs_values.norm(norm).item()

                    layer_stats[name] = stats
        self.log_stats(self.epoch, batch_number, logs, layer_stats)

    def log_stats(
        self, epoch_number: int, batch_number: int, logs: Dict, layer_stats: Dict[str, Dict[str, float]]
    ) -> None:
        raise NotImplementedError

    def _keep_layer(self, param: torch.nn.parameter.Parameter, name: str):
        if self.keep_bias:
            return param.requires_grad
        return param.requires_grad and ("bias" not in name)


class MemoryGradientLogger(GradientLoggerBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.history = []

    def on_train_begin(self, logs: Dict):
        super().on_train_begin(logs)
        self.history = {layer: [] for layer in self.layers}

    def log_stats(
        self, epoch_number: int, batch_number: int, logs: Dict, layer_stats: Dict[str, Dict[str, float]]
    ) -> None:
        for layer, stats in layer_stats.items():
            stats['epoch'] = epoch_number
            stats['batch'] = batch_number
            self.history[layer].append(stats)


class TensorBoardGradientLogger(GradientLoggerBase):
    def __init__(self, writer, initial_step: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.writer = writer
        self.current_step = initial_step

    def log_stats(
        self, epoch_number: int, batch_number: int, logs: Dict, layer_stats: Dict[str, Dict[str, float]]
    ) -> None:
        self.current_step += 1
        for layer, stats in layer_stats.items():
            for name, value in stats.items():
                self.writer.add_scalars(f'gradient_stats/{layer}', {name: value}, self.current_step)


class AtomicCSVGradientLogger(GradientLoggerBase):
    def __init__(
        self,
        filename,
        *,
        separator: str = ',',
        append: bool = False,
        temporary_filename: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.filename = filename
        self.temporary_filename = temporary_filename
        self.separator = separator
        self.append = append
        self.fieldnames = []

    def _save_stats(self, fd: TextIO, filename: str, stats: Dict):
        olddata = None
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as oldfile:
                olddata = list(csv.DictReader(oldfile, delimiter=self.separator))
        csvwriter = csv.DictWriter(fd, fieldnames=self.fieldnames, delimiter=self.separator)
        csvwriter.writeheader()
        if olddata is not None:
            csvwriter.writerows(olddata)
        csvwriter.writerow(stats)

    def _write_header(self, fd: TextIO):
        csvwriter = csv.DictWriter(fd, fieldnames=self.fieldnames, delimiter=self.separator)
        csvwriter.writeheader()

    def on_train_begin(self, logs: Dict):
        super().on_train_begin(logs)
        self.fieldnames = ['epoch', 'batch'] + self.stats_names

        if not self.append:
            for layer in self.layers:
                filename = self.filename.format(layer)
                atomic_lambda_save(filename, self._write_header, (), temporary_filename=self.temporary_filename)

    def log_stats(
        self, epoch_number: int, batch_number: int, logs: Dict, layer_stats: Dict[str, Dict[str, float]]
    ) -> None:
        for layer, stats in layer_stats.items():
            filename = self.filename.format(layer)
            stats['epoch'] = epoch_number
            stats['batch'] = batch_number
            atomic_lambda_save(
                filename, self._save_stats, (filename, stats), temporary_filename=self.temporary_filename
            )


class CSVGradientLogger(GradientLoggerBase):
    def __init__(self, filename, *, separator: str = ',', append: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.filename = filename
        self.separator = separator
        self.append = append

    def on_train_begin(self, logs: Dict):
        super().on_train_begin(logs)
        fieldnames = ['epoch', 'batch'] + self.stats_names
        open_flag = 'a' if self.append else 'w'

        self.csvfiles = {}
        self.writers = {}
        for layer in self.layers:
            filename = self.filename.format(layer)
            self.csvfiles[layer] = open(filename, open_flag, newline='', encoding='utf-8')
            self.writers[layer] = csv.DictWriter(self.csvfiles[layer], fieldnames=fieldnames, delimiter=self.separator)
            if not self.append:
                self.writers[layer].writeheader()
                self.csvfiles[layer].flush()

    def log_stats(
        self, epoch_number: int, batch_number: int, logs: Dict, layer_stats: Dict[str, Dict[str, float]]
    ) -> None:
        for layer, stats in layer_stats.items():
            stats['epoch'] = epoch_number
            stats['batch'] = batch_number
            self.writers[layer].writerow(stats)
            self.csvfiles[layer].flush()

    def on_train_end(self, logs: Dict):
        super().on_train_end(logs)
        for layer in self.layers:
            self.csvfiles[layer].close()
