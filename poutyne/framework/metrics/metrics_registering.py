from .utils import camel_to_snake


def _get_registering_decorator(register_function):
    def register(name_or_func, *extra_names):
        if isinstance(name_or_func, str):
            names = [name_or_func] + list(extra_names)

            def decorator_func(func):
                register_function(func, names)
                return func

            return decorator_func

        func = name_or_func
        register_function(func)
        return func

    return register


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


register_batch_metric = _get_registering_decorator(register_batch_metric_function)


def get_loss_or_metric(loss_metric):
    if isinstance(loss_metric, str):
        loss_metric = clean_batch_metric_name(loss_metric)
        return batch_metrics_dict[loss_metric]

    return loss_metric


epochs_metrics_dict = {}


def clean_epoch_metric_name(name):
    name = name.lower()
    name = name[:-5] if name.endswith('score') else name
    name = name.replace('_', '')
    return name


def register_epoch_metric_class(clz, names=None):
    names = [camel_to_snake(clz.__name__)] if names is None else names
    names = [names] if isinstance(names, str) else names
    epochs_metrics_dict.update({clean_epoch_metric_name(name): clz for name in names})


register_epoch_metric = _get_registering_decorator(register_epoch_metric_class)


def get_epoch_metric(epoch_metric):
    if isinstance(epoch_metric, str):
        epoch_metric = clean_epoch_metric_name(epoch_metric)
        return epochs_metrics_dict[epoch_metric]()
    return epoch_metric
