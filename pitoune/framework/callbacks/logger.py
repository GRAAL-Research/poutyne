import csv
from .callbacks import Callback

class CSVLogger(Callback):
    def __init__(self, filename, separator=','):
        self.filename = filename
        self.separator = separator

    def on_train_begin(self, logs=None):
        metrics = ['loss'] + self.model.metrics_names
        fieldnames = ['epoch', 'lr'] + metrics + ['val_' + metric for metric in metrics]
        self.csvfile = open(self.filename, 'w', newline='')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=fieldnames, delimiter=self.separator)
        self.writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        self.writer.writerow(logs[-1])

    def on_train_end(self, logs=None):
        self.csvfile.close()
