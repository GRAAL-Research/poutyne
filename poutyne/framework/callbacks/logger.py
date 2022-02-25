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

import csv
import os
from typing import Dict, Optional, TextIO

from ._utils import atomic_lambda_save
from .callbacks import Callback


class Logger(Callback):
    def __init__(self, *, batch_granularity: bool = False):
        super().__init__()
        self.batch_granularity = batch_granularity
        self.epoch_number = 0

    def on_train_begin(self, logs: Dict):
        metrics = ['loss'] + self.model.metrics_names

        if self.batch_granularity:
            self.fieldnames = ['epoch', 'batch', 'size', 'time']
        else:
            self.fieldnames = ['epoch', 'time']
        if getattr(self.model, 'optimizer', None) is not None:
            if len(self.model.optimizer.param_groups) > 1:
                self.fieldnames += [f'lr_group_{i}' for i in range(len(self.model.optimizer.param_groups))]
            else:
                self.fieldnames += ['lr']
        self.fieldnames += metrics
        self.fieldnames += ['val_' + metric for metric in metrics]
        self._on_train_begin_write(logs)

    def _on_train_begin_write(self, logs: Dict):
        pass

    def on_train_batch_end(self, batch_number: int, logs: Dict):
        if self.batch_granularity:
            logs = self._get_logs_without_unknown_keys(logs)
            self._on_train_batch_end_write(batch_number, logs)

    def _on_train_batch_end_write(self, batch_number: int, logs: Dict):
        pass

    def on_epoch_begin(self, epoch_number: int, logs: Dict):
        self.epoch_number = epoch_number
        self._on_epoch_begin_write(self.epoch_number, logs)

    def _on_epoch_begin_write(self, epoch_number: int, logs: Dict):
        pass

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        logs = self._get_logs_without_unknown_keys(logs)
        self._on_epoch_end_write(epoch_number, logs)

    def _on_epoch_end_write(self, epoch_number: int, logs: Dict):
        pass

    def on_train_end(self, logs: Dict):
        self._on_train_end_write(logs)

    def _on_train_end_write(self, logs: Dict):
        pass

    def _get_logs_without_unknown_keys(self, logs: Dict):
        return {k: logs[k] for k in self.fieldnames if logs.get(k) is not None}

    def _get_current_learning_rates(self):
        learning_rates = {}
        if getattr(self.model, 'optimizer', None) is not None:
            if len(self.model.optimizer.param_groups) > 1:
                learning_rates = {
                    f'lr_group_{i}': param_group['lr']
                    for i, param_group in enumerate(self.model.optimizer.param_groups)
                }
            else:
                learning_rates = {'lr': self.model.optimizer.param_groups[0]['lr']}
        return learning_rates


class CSVLogger(Logger):
    """
    Callback that outputs the result of each epoch_number or batch into a CSV file.

    Args:
        filename (str): The filename of the CSV.
        batch_granularity (bool): Whether to also output the result of each batch in addition to the epochs.
            (Default value = False)
        separator (str): The separator to use in the CSV.
            (Default value = ',')
        append (bool): Whether to append to an existing file.

    """

    def __init__(self, filename: str, *, batch_granularity: bool = False, separator: str = ',', append: bool = False):
        super().__init__(batch_granularity=batch_granularity)
        self.filename = filename
        self.separator = separator
        self.append = append

    def _on_train_begin_write(self, logs: Dict):
        open_flag = 'a' if self.append else 'w'
        self.csvfile = open(self.filename, open_flag, newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames, delimiter=self.separator)
        if not self.append:
            self.writer.writeheader()
            self.csvfile.flush()

    def _on_train_batch_end_write(self, batch_number: int, logs: Dict):
        self.writer.writerow(logs)
        self.csvfile.flush()

    def _on_epoch_end_write(self, epoch_number: int, logs: Dict):
        self.writer.writerow({**logs, **self._get_current_learning_rates()})
        self.csvfile.flush()

    def _on_train_end_write(self, logs: Dict):
        self.csvfile.close()


class AtomicCSVLogger(Logger):
    """
    Callback that outputs the result of each epoch_number or batch into a CSV file in an atomic matter.

    Args:
        filename (str): The filename of the CSV.
        temporary_filename (str, optional): Temporary filename for the CSV file so that it can be written
            atomically.
        batch_granularity (bool): Whether to also output the result of each batch in addition to the epochs.
            (Default value = False)
        separator (str): The separator to use in the CSV.
            (Default value = ',')
        append (bool): Whether to append to an existing file.
    """

    def __init__(
        self,
        filename,
        *,
        batch_granularity: bool = False,
        separator: str = ',',
        append: bool = False,
        temporary_filename: Optional[str] = None,
    ):
        super().__init__(batch_granularity=batch_granularity)
        self.filename = filename
        self.temporary_filename = temporary_filename
        self.separator = separator
        self.append = append

    def _save_log(self, fd: TextIO, logs: Dict):
        olddata = None
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as oldfile:
                olddata = list(csv.DictReader(oldfile, delimiter=self.separator))
        csvwriter = csv.DictWriter(fd, fieldnames=self.fieldnames, delimiter=self.separator)
        csvwriter.writeheader()
        if olddata is not None:
            csvwriter.writerows(olddata)
        if logs is not None:
            csvwriter.writerow(logs)

    def _write_header(self, fd: TextIO):
        csvwriter = csv.DictWriter(fd, fieldnames=self.fieldnames, delimiter=self.separator)
        csvwriter.writeheader()

    def _on_train_begin_write(self, logs: Dict):
        if not self.append:
            atomic_lambda_save(self.filename, self._write_header, (), temporary_filename=self.temporary_filename)

    def _on_train_batch_end_write(self, batch_number: int, logs: Dict):
        atomic_lambda_save(self.filename, self._save_log, (logs,), temporary_filename=self.temporary_filename)

    def _on_epoch_end_write(self, epoch_number: int, logs: Dict):
        logs = {**logs, **self._get_current_learning_rates()}
        atomic_lambda_save(self.filename, self._save_log, (logs,), temporary_filename=self.temporary_filename)


class TensorBoardLogger(Logger):
    """
    Callback that outputs the result of each epoch_number or batch into a Tensorboard experiment folder.

    Args:
        writer (~torch.utils.tensorboard.writer.SummaryWriter): The tensorboard writer.
        split_train_val (bool): Whether to put each training and validation metric in the same graphs.
            (Default = False)

    Example:
        Using TensorBoardLogger::

            from torch.utils.tensorboard import SummaryWriter
            from poutyne import Model, TensorBoardLogger

            writer = SummaryWriter('runs')
            tb_logger = TensorBoardLogger(writer)

            model = Model(...)
            model.fit_generator(..., callbacks=[tb_logger])
    """

    def __init__(self, writer, split_train_val: bool = False):
        super().__init__(batch_granularity=False)
        self.writer = writer
        self.split_train_val = split_train_val

    def _on_train_batch_end_write(self, batch_number: int, logs):
        """
        We don't handle tensorboard writing on batch granularity
        """
        pass

    def _on_epoch_end_write(self, epoch_number: int, logs: dict):
        lr = self._get_current_learning_rates()

        if not self.split_train_val:
            grouped_items = {}
            for k, v in logs.items():
                if 'val_' in k:
                    primary_key = k[4:]
                    if primary_key not in grouped_items:
                        grouped_items[primary_key] = {}
                    grouped_items[k[4:]][k] = v
                else:
                    if k not in grouped_items:
                        grouped_items[k] = {}
                    grouped_items[k][k] = v

            for k, v in grouped_items.items():
                self.writer.add_scalars(k, v, epoch_number)
            self.writer.add_scalars('lr', lr, epoch_number)
        else:
            metrics = {**logs, **lr}
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(metric_name, metric_value, epoch_number)

    def _on_train_end_write(self, logs: Dict):
        self.writer.close()
