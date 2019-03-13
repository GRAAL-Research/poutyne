import csv
from .callbacks import Callback


class Logger(Callback):
    def __init__(self, *, batch_granularity=False):
        super().__init__()
        self.batch_granularity = batch_granularity
        self.epoch = 0

    def on_train_begin(self, logs):
        metrics = ['loss'] + self.model.metrics_names

        if self.batch_granularity:
            self.fieldnames = ['epoch', 'batch', 'size', 'time', 'lr']
        else:
            self.fieldnames = ['epoch', 'time', 'lr']
        self.fieldnames += metrics
        self.fieldnames += ['val_' + metric for metric in metrics]
        self._on_train_begin_write(logs)

    def _on_train_begin_write(self, logs):
        pass

    def on_batch_end(self, batch, logs):
        if self.batch_granularity:
            logs = self._get_logs_without_unknown_keys(logs)
            self._on_batch_end_write(batch, logs)

    def _on_batch_end_write(self, batch, logs):
        pass

    def on_epoch_begin(self, epoch, logs):
        self.epoch = epoch
        self._on_epoch_begin_write(epoch, logs)

    def _on_epoch_begin_write(self, epoch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        logs = self._get_logs_without_unknown_keys(logs)
        self._on_epoch_end_write(epoch, logs)

    def _on_epoch_end_write(self, epoch, logs):
        pass

    def on_train_end(self, logs=None):
        self._on_train_end_write(logs)

    def _on_train_end_write(self, logs):
        pass

    def _get_logs_without_unknown_keys(self, logs):
        return {k:logs[k] for k in self.fieldnames if logs.get(k) is not None}

    def _get_current_learning_rates(self):
        learning_rates = [param_group['lr'] for param_group in self.model.optimizer.param_groups]
        return learning_rates[0] if len(learning_rates) == 1 else learning_rates


class CSVLogger(Logger):
    """
    Callback that output the result of each epoch or batch into a CSV file.

    Args:
        filename (string): The filename of the CSV.
        batch_granularity (bool): Whether to also output the result of each
            batch in addition to the epochs. (Default value = False)
        separator (string): The separator to use in the CSV.
            (Default value = ',')
        append (bool): Whether to append to an existing file.

    """
    def __init__(self, filename, *, batch_granularity=False, separator=',', append=False):
        super().__init__(batch_granularity=batch_granularity)
        self.filename = filename
        self.separator = separator
        self.append = append

    def _on_train_begin_write(self, logs):
        open_flag = 'a' if self.append else 'w'
        self.csvfile = open(self.filename, open_flag, newline='')
        self.writer = csv.DictWriter(self.csvfile,
                                     fieldnames=self.fieldnames,
                                     delimiter=self.separator)
        if not self.append:
            self.writer.writeheader()
            self.csvfile.flush()

    def _on_batch_end_write(self, batch, logs):
        self.writer.writerow(logs)
        self.csvfile.flush()

    def _on_epoch_end_write(self, epoch, logs):
        self.writer.writerow(dict(logs, lr=self._get_current_learning_rates()))
        self.csvfile.flush()

    def _on_train_end_write(self, logs=None):
        self.csvfile.close()


class TensorBoardLogger(Logger):
    """
    Callback that output the result of each epoch or batch into a Tensorboard experiment folder.

    Args:
        writer (tensorboardX.SummaryWriter): The tensorboard writer.

    Example:
        Using tensorboardX::

            from tensorboardX import SummaryWriter
            from pytoune.framework import Model
            from pytoune.framework.callbacks import TensorBoardLogger

            writer = SummaryWriter('runs')
            tb_logger = TensorBoardLogger(writer)

            model = Model(...)
            model.fit_generator(..., callbacks=[tb_logger])
    """
    def __init__(self, writer):
        super().__init__(batch_granularity=False)
        self.writer = writer

    def _on_batch_end_write(self, batch, logs):
        """
        We don't handle tensorboard writing on batch granularity
        """
        pass

    def _on_epoch_end_write(self, epoch, logs):
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
            self.writer.add_scalars(k, v, epoch)
        lr = self._get_current_learning_rates()
        if isinstance(lr, (list,)):
            self.writer.add_scalars(
                'lr',
                {str(i): v for i, v in enumerate(lr)},
                epoch
            )
        else:
            self.writer.add_scalars('lr', {'lr': lr}, epoch)
