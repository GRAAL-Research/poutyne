from torch.nn import CrossEntropyLoss


class SequenceCrossEntropy:
    """
    Wrapper around the cross entropy loss for a padded sequence
    """

    def __init__(self, padding_value=0):
        self.loss_fct = CrossEntropyLoss(ignore_index=padding_value)

    def __call__(self, y_pred, y):
        """
        Flatten the sequence before passing it to the CrossEntropyLoss.

        Args:
            y_pred (~torch.Tensor): The prediction from a model.
            y (~torch.Tensor or Union[~torch.Tensor, ~torch.Tensor]): The target.
                ~torch.Tensor if only the target.
                Union[~torch.Tensor, ~torch.Tensor] if the target and a mask used to mask the padded elements.

        Returns:
                The loss value
        """
        if isinstance(y, tuple):
            y = y[0]

        flatten_y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)
        flatten_y = y.view(y.shape[0] * y.shape[1])
        return self.loss_fct(flatten_y_pred, flatten_y)
