
batch_metrics_dict = {}


def clean_batch_metric_name(name):
    name = name.lower()
    name = name[:-4] if name.endswith('loss') else name
    name = name.replace('_', '')
    return name


def register_batch_metric_function(func, names=None):
    names = [func.__name__] if names is None else names
    names = [names] if isinstance(names, str) else names
    batch_metrics_dict.update({clean_batch_metric_name(name): func for name in names})


def register_batch_metric(name_or_func, *extra_names):
    if isinstance(name_or_func, str):
        names = [name_or_func] + list(extra_names)
        def decorator_func(func):
            register_batch_metric_function(func, names)
            return func
        return decorator_func

    func = name_or_func
    register_batch_metric_function(func)
    return func


def get_loss_or_metric(loss_metric):
    if isinstance(loss_metric, str):
        loss_metric = clean_batch_metric_name(loss_metric)
        return batch_metrics_dict[loss_metric]

    return loss_metric
