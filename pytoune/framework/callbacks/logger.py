import csv
from .callbacks import Callback

class CSVLogger(Callback):
    """
    Callback that output the result of each epoch or batch into a CSV file.

    Args:
        filename (string): The filename of the CSV.
        batch_granularity (bool): Whether to also output the result of each
            batch in addition to the epochs. (Default value = False)
        separator (string): The separator to use in the CSV.
            (Default value = ',')
        append (bool): Whether to append to an existing file.
            (Default value = False)
    """
    def __init__(self, filename, batch_granularity=False, separator=',', append=False):
        self.filename = filename
        self.batch_granularity = batch_granularity
        self.separator = separator
        self.append = append

    def on_train_begin(self, logs):
        metrics = ['loss'] + self.model.metrics_names

        if self.batch_granularity:
            self.fieldnames = ['epoch', 'batch', 'size', 'lr']
        else:
            self.fieldnames = ['epoch', 'lr']
        self.fieldnames += metrics
        self.fieldnames += ['val_' + metric for metric in metrics]

        open_flag = 'a' if self.append else 'w'
        self.csvfile = open(self.filename, open_flag, newline='')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames, delimiter=self.separator)
        if not self.append:
            self.writer.writeheader()
            self.csvfile.flush()

    def on_batch_end(self, batch, logs):
        if self.batch_granularity:
            logs = self._get_logs_without_unknown_keys(logs)
            self.writer.writerow(logs)
            self.csvfile.flush()

    def on_epoch_end(self, epoch, logs):
        logs = self._get_logs_without_unknown_keys(logs)
        self.writer.writerow(dict(logs, lr=self._get_current_learning_rates()))
        self.csvfile.flush()

    def _get_logs_without_unknown_keys(self, logs):
        return {k:logs[k] for k in self.fieldnames if logs.get(k)}

    def _get_current_learning_rates(self):
        learning_rates = [param_group['lr'] for param_group in self.model.optimizer.param_groups]
        return learning_rates[0] if len(learning_rates) == 1 else learning_rates

    def on_train_end(self, logs=None):
        self.csvfile.close()
