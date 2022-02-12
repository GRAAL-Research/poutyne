from .utils import camel_to_snake


def _get_registering_decorator(register_function):
    def decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and len(kwargs) == 0:
            register_function(args[0])
            return args[0]

        def register(func):
            register_function(func, args, **kwargs)
            return func

        return register

    return decorator


batch_metrics_dict = {}


def clean_batch_metric_name(name):
    name = name.lower()
    name = name[:-4] if name.endswith('loss') else name
    name = name.replace('_', '')
    return name


def register_batch_metric_function(func, names=None, unique_name=None):
    names = [func.__name__] if names is None or len(names) == 0 else names
    names = [names] if isinstance(names, str) else names
    names = [clean_batch_metric_name(name) for name in names]
    if unique_name is None:
        update = {name: func for name in names}
    else:
        update = {name: (unique_name, func) for name in names}
    batch_metrics_dict.update(update)
    return names


register_batch_metric = _get_registering_decorator(register_batch_metric_function)


def get_loss_or_metric(loss_metric):
    if isinstance(loss_metric, str):
        loss_metric = clean_batch_metric_name(loss_metric)
        return batch_metrics_dict[loss_metric]
    if isinstance(loss_metric, tuple) and isinstance(loss_metric[1], str):
        name, loss_metric = loss_metric
        loss_metric = clean_batch_metric_name(loss_metric)
        loss_metric = batch_metrics_dict[loss_metric]
        if isinstance(loss_metric, tuple):
            loss_metric = loss_metric[1]
        return name, loss_metric
    return loss_metric


epochs_metrics_dict = {}


def clean_epoch_metric_name(name):
    name = name.lower()
    name = name[:-5] if name.endswith('score') else name
    name = name.replace('_', '')
    return name


def register_epoch_metric_class(clz, names=None, unique_name=None):
    names = [camel_to_snake(clz.__name__)] if names is None or len(names) == 0 else names
    names = [names] if isinstance(names, str) else names
    names = [clean_epoch_metric_name(name) for name in names]
    if unique_name is None:
        update = {name: clz for name in names}
    else:
        update = {name: (unique_name, clz) for name in names}
    epochs_metrics_dict.update(update)
    return names


def unregister_epoch_metric(names):
    for name in names:
        del epochs_metrics_dict[name]


register_epoch_metric = _get_registering_decorator(register_epoch_metric_class)


def get_epoch_metric(epoch_metric):
    if isinstance(epoch_metric, str):
        epoch_metric = clean_epoch_metric_name(epoch_metric)
        epoch_metric = epochs_metrics_dict[epoch_metric]
        if isinstance(epoch_metric, tuple):
            name, epoch_metric = epoch_metric
            return name, epoch_metric()
        return epoch_metric()
    if isinstance(epoch_metric, tuple) and isinstance(epoch_metric[1], str):
        name, epoch_metric = epoch_metric
        epoch_metric = clean_epoch_metric_name(epoch_metric)
        epoch_metric = epochs_metrics_dict[epoch_metric]
        if isinstance(epoch_metric, tuple):
            epoch_metric = epoch_metric[1]
        return name, epoch_metric()
    return epoch_metric
