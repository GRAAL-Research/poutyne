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
            self.fieldnames = ['epoch', 'batch', 'size', 'time', 'lr']
        else:
            self.fieldnames = ['epoch', 'time', 'lr']
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
        learning_rates = [param_group['lr'] for param_group in self.model.optimizer.param_groups]
        return learning_rates[0] if len(learning_rates) == 1 else learning_rates


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
        self.csvfile = open(self.filename, open_flag, newline='')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames, delimiter=self.separator)
        if not self.append:
            self.writer.writeheader()
            self.csvfile.flush()

    def _on_train_batch_end_write(self, batch_number: int, logs: Dict):
        self.writer.writerow(logs)
        self.csvfile.flush()

    def _on_epoch_end_write(self, epoch_number: int, logs: Dict):
        self.writer.writerow(dict(logs, lr=self._get_current_learning_rates()))
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

    def __init__(self,
                 filename,
                 *,
                 batch_granularity: bool = False,
                 separator: str = ',',
                 append: bool = False,
                 temporary_filename: Optional[str] = None):
        super().__init__(batch_granularity=batch_granularity)
        self.filename = filename
        self.temporary_filename = temporary_filename
        self.separator = separator
        self.append = append

    def _save_log(self, fd: TextIO, logs: Dict):
        olddata = None
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as oldfile:
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
        atomic_lambda_save(self.filename, self._save_log, (logs, ), temporary_filename=self.temporary_filename)

    def _on_epoch_end_write(self, epoch_number: int, logs: Dict):
        logs = dict(logs, lr=self._get_current_learning_rates())
        atomic_lambda_save(self.filename, self._save_log, (logs, ), temporary_filename=self.temporary_filename)


class TensorBoardLogger(Logger):
    """
    Callback that outputs the result of each epoch_number or batch into a Tensorboard experiment folder.

    Args:
        writer (~torch.utils.tensorboard.writer.SummaryWriter): The tensorboard writer.

    Example:
        Using TensorBoardLogger::

            from torch.utils.tensorboard import SummaryWriter
            from poutyne.framework import Model
            from poutyne.framework.callbacks import TensorBoardLogger

            writer = SummaryWriter('runs')
            tb_logger = TensorBoardLogger(writer)

            model = Model(...)
            model.fit_generator(..., callbacks=[tb_logger])
    """

    def __init__(self, writer):
        super().__init__(batch_granularity=False)
        self.writer = writer

    def _on_train_batch_end_write(self, batch_number: int, logs):
        """
        We don't handle tensorboard writing on batch granularity
        """
        pass

    def _on_epoch_end_write(self, epoch_number: int, logs: dict):
        grouped_items = dict()
        for k, v in logs.items():
            if 'val_' in k:
                primary_key = k[4:]
                if primary_key not in grouped_items:
                    grouped_items[primary_key] = dict()
                grouped_items[k[4:]][k] = v
            else:
                if k not in grouped_items:
                    grouped_items[k] = dict()
                grouped_items[k][k] = v
        for k, v in grouped_items.items():
            self.writer.add_scalars(k, v, epoch_number)
        lr = self._get_current_learning_rates()
        if isinstance(lr, (list, )):
            self.writer.add_scalars('lr', {str(i): v for i, v in enumerate(lr)}, epoch_number)
        else:
            self.writer.add_scalars('lr', {'lr': lr}, epoch_number)

    def _on_train_end_write(self, logs: Dict):
        self.writer.close()
