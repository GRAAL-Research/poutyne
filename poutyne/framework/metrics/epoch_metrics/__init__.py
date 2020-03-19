# pylint: disable=wildcard-import
from .base import *
from .fscores import *
from .sklearn_metrics import *

all_epochs_metrics_dict = dict(f1=F1)


def get_epoch_metric(epoch_metric):
    if isinstance(epoch_metric, str):
        epoch_metric = epoch_metric.lower()
        epoch_metric = epoch_metric[:-5] if epoch_metric.endswith('score') else epoch_metric
        epoch_metric = epoch_metric.replace('_', '')
        return all_epochs_metrics_dict[epoch_metric]()
    return epoch_metric
