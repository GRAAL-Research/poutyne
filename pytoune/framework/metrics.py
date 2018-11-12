import torch.nn.functional as F

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

all_losses_metrics_dict = dict(
    acc=acc,
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
    softmargin=soft_margin
)

def get_loss_or_metric(loss_metric):
    if isinstance(loss_metric, str):
        loss_metric = loss_metric.lower()
        loss_metric = loss_metric[:-4] if loss_metric.endswith('loss') else loss_metric
        loss_metric = loss_metric.replace('_', '')
        return all_losses_metrics_dict[loss_metric]

    return loss_metric
