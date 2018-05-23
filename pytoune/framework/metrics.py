
def acc(y_pred, y_true):
    _, y_pred = y_pred.max(1)
    acc_pred = (y_pred == y_true).float().mean()
    return acc_pred * 100

all_metrics_dict = dict(
    acc=acc,
    accuracy=acc
)

def get_metric(metric):
    if isinstance(metric, str):
        return all_metrics_dict[metric]
    else:
        return metric
