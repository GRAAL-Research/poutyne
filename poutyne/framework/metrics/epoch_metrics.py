import sys

from abc import ABC, abstractmethod

try:
    import allennlp.training.metrics.fbeta_measure as fbeta_measure
except ImportError:
    fbeta_measure = None


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
    def get_metric(self):
        """
        Compute and return the metric.
        """
        pass


class FBeta(EpochMetric):
    """
    Wrapper around the Allen NLP FBetaMeasure class.

    Args:
        beta : (float), optional (default = 1.0) The strength of recall versus precision in the F-score.

        average : (str) or (int), ['micro' (default), 'macro', label_number]
        If the argument is of type integer, the score for this class (the label number) is calculated. Otherwise, this
        determines the type of averaging performed on all the data:
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted mean.
            This does not take label imbalance into account.

    """

    def __init__(self, beta=1.0, average='micro'):
        super().__init__()
        if ".".join([str(sys.version_info[0]), str(sys.version_info[1]), str(sys.version_info[2])]) < "3.6.1":
            raise NotImplementedError("allen nlp don't support python version older than 3.6.1.")
        if fbeta_measure is None:
            raise ImportError("allen nlp need to be installed to use this class.")

        self.label = None
        if isinstance(average, int):
            self.label = average
            self.running_measure = fbeta_measure.FBetaMeasure(beta=beta)
        else:
            self.running_measure = fbeta_measure.FBetaMeasure(beta=beta, average=average)

    def __call__(self, y_prediction, y_true):
        """
        Update the confusion matrix for calculating the F-score.

        Args:
            y_predict : Predictions of the model.
            y_true : A tensor of the gold labels. Can also be a tuple of gold_label and a mask.
        """

        mask = None
        if isinstance(y_true, tuple):
            y_true, mask = y_true
        self.running_measure(y_prediction, y_true, mask=mask)

    def get_metric(self):
        """
        Method to get the metric score.

        """

        if self.label is not None:
            return self.running_measure.get_metric(reset=True)['fscore'][self.label]
        return self.running_measure.get_metric(reset=True)['fscore']


class F1(FBeta):
    """
    Wrapper around the Allen NLP FBetaMeasure class in the specific case where beta is equal to one (1).

    Args:
        average : (str) or (int), ['micro' (default), 'macro', label_number]
        If the arguments is of type integer, The score for this class (the label number) is calculated. Otherwise, this
        determines the type of averaging performed on all the data:
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted mean.
            This does not take label imbalance into account.

    Attributes:
        Average (str) or (int): The method to calculate the F-score.

    """

    def __init__(self, average='micro'):
        super().__init__(beta=1, average=average)


all_epochs_metrics_dict = dict(f1=F1)


def get_epoch_metric(epoch_metric):
    if isinstance(epoch_metric, str):
        epoch_metric = epoch_metric.lower()
        epoch_metric = epoch_metric[:-5] if epoch_metric.endswith('score') else epoch_metric
        epoch_metric = epoch_metric.replace('_', '')
        return all_epochs_metrics_dict[epoch_metric]
    return epoch_metric
