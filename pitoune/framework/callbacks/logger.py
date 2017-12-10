import csv
from .callbacks import Callback

class CSVLogger(Callback):
    def __init__(self, filename, separator=','):
        self.filename = filename
        self.separator = separator

    def on_train_begin(self, logs=None):
        self.metrics = ['loss'] + self.params['metrics']

    def on_train_end(self, logs=None):
        with open(self.filename, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'lr'] + self.metrics + ['val_' + metric for metric in self.metrics]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=self.separator)
            writer.writeheader()
            writer.writerows(logs)
