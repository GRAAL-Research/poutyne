from abc import ABC, abstractmethod

try:
    from allennlp.training.metrics import FBetaMeasure
except ImportError:
    FBetaMeasure = None


class EpochMetric(ABC):
    # pylint: disable=line-too-long
    """
    The abstract class representing a epoch metric which can be accumulated at each batch and calculated at the end
    of the epoch.
    """

    @abstractmethod
    def __call__(self, y_prediction, y_true):
        """
        To define the behavior of the metric when called.

        Args:
            y_prediction: The prediction of the model.
            y_true: Target to evaluate the model.
        """
        pass

    @abstractmethod
    def __init__(self):
        self.__name__ = self.__class__.__name__

    @abstractmethod
    def get_metric(self, reset):
        """
        Compute and return the metric. Optionally also reset the running measure.
        """
        pass


class F1(EpochMetric):
    """
    Wrapper around the Allen NLP FBetaMeasure class.
    average can be 'micro, macro or none'

    """

    def __init__(self, beta=1.0, average='micro'):
        super().__init__()
        self.average = average  # to verify if we can extract a F1 score for a specific label with get_metric method.
        if FBetaMeasure is None:
            raise ImportError("allen nlp need to be installed to use this class.")
        self.running_measure = FBetaMeasure(beta=beta, average=average)

    def __call__(self, y_prediction, y_true):
        mask = None
        if isinstance(y_true, tuple):
            y_true, mask = y_true
        self.running_measure(y_prediction, y_true, mask=mask)

    def get_metric(self, reset=True):
        return self.running_measure.get_metric(reset=reset)['fscore']


all_epochs_metrics_dict = dict(f1=F1)


def get_epoch_metric(epoch_metric):
    if isinstance(epoch_metric, str):
        epoch_metric = epoch_metric.lower()
        epoch_metric = epoch_metric[:-5] if epoch_metric.endswith('score') else epoch_metric
        epoch_metric = epoch_metric.replace('_', '')
        return all_epochs_metrics_dict[epoch_metric]
    return epoch_metric
