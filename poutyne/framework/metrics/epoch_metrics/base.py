from abc import ABC, abstractmethod

import torch.nn as nn


class EpochMetric(ABC, nn.Module):
    """
    The abstract class representing a epoch metric which can be accumulated at each batch and calculated at the end
    of the epoch.
    """
    @abstractmethod
    def forward(self, y_pred, y_true):
        """
        To define the behavior of the metric when called.

        Args:
            y_pred: The prediction of the model.
            y_true: Target to evaluate the model.
        """
        pass

    @abstractmethod
    def get_metric(self):
        """
        Compute and return the metric.
        """
        pass
