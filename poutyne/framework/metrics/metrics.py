import torch.nn.functional as F
from torch import tensor


def acc(y_pred, y_true, ignore_index=-100):
    y_pred = y_pred.argmax(1)
    weights = (y_true != ignore_index).float()
    num_labels = weights.sum()
    acc_pred = ((y_pred == y_true).float() * weights).sum() / num_labels
    return acc_pred * 100


def bin_acc(y_pred, y_true, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100


def bce(y_pred, y_true):
    return F.binary_cross_entropy(y_pred, y_true)


def poisson_nll(y_pred, y_true):
    return F.poisson_nll_loss(y_pred, y_true)


def hinge_embedding(y_pred, y_true):
    return F.hinge_embedding_loss(y_pred, y_true)


def l1(y_pred, y_true):
    return F.l1_loss(y_pred, y_true)


def mse(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)


def multilabel_margin(y_pred, y_true):
    return F.multilabel_margin_loss(y_pred, y_true)


def multilabel_soft_margin(y_pred, y_true):
    return F.multilabel_soft_margin_loss(y_pred, y_true)


def multi_margin(y_pred, y_true):
    return F.multi_margin_loss(y_pred, y_true)


def nll(y_pred, y_true):
    return F.nll_loss(y_pred, y_true)


def bce_with_logits(y_pred, y_true):
    return F.binary_cross_entropy_with_logits(y_pred, y_true)


def smooth_l1(y_pred, y_true):
    return F.smooth_l1_loss(y_pred, y_true)


def soft_margin(y_pred, y_true):
    return F.soft_margin_loss(y_pred, y_true)


def sequence_acc(y_pred: tensor, y: tuple):
    # pylint: disable=line-too-long
    """
    The masked accuracy for sequence. The mask is use to *remove* the padded value and not calculate the accuracy over these value.

    Args:
        y_pred (~torch.Tensor): The prediction from a model.
        y (Union[~torch.Tensor, ~torch.Tensor]): The target and a mask used to remove the padded element in the calculation.

    Returns:
        A float accuracy value.
    """
    y_pred = y_pred.view(y_pred.shape[0] * y_pred.shape[1], -1)

    if not isinstance(y, tuple):
        raise TypeError("The y_pred argument must be a tuple of prediction and the masked of the padding elements..")

    y_true, mask = y
    y_true = y_true.view(y_true.shape[0] * y_true.shape[1])
    y_true = y_true.float()
    mask = mask.view(mask.shape[0] * mask.shape[1])

    argmax_y_pred = y_pred.max(dim=1)[1].float()
    true_positives = (y_true == argmax_y_pred) * mask

    true_positive_sum = y_true[true_positives].shape[0]
    pred_sum = argmax_y_pred[mask].long().shape[0]

    precision = true_positive_sum / pred_sum * 100
    return precision


all_losses_metrics_dict = dict(acc=acc,
                               accuracy=acc,
                               binacc=bin_acc,
                               binaryacc=bin_acc,
                               binaryaccuracy=bin_acc,
                               binarycrossentropy=bce,
                               bce=bce,
                               poissonnll=poisson_nll,
                               crossentropy=F.cross_entropy,
                               hingeembedding=hinge_embedding,
                               kldiv=F.kl_div,
                               l1=l1,
                               mse=mse,
                               multilabelmargin=multilabel_margin,
                               multilabelsoftmargin=multilabel_soft_margin,
                               multimargin=multi_margin,
                               nll=nll,
                               binarycrossentropywithlogits=bce_with_logits,
                               bcewithlogits=bce_with_logits,
                               smoothl1=smooth_l1,
                               softmargin=soft_margin,
                               sequenceacc=sequence_acc,
                               seqacc=sequence_acc)


def get_loss_or_metric(loss_metric):
    if isinstance(loss_metric, str):
        loss_metric = loss_metric.lower()
        loss_metric = loss_metric[:-4] if loss_metric.endswith('loss') else loss_metric
        loss_metric = loss_metric.replace('_', '')
        return all_losses_metrics_dict[loss_metric]

    return loss_metric
