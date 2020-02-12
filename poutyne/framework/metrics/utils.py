import re
from collections import Counter

# From https://stackoverflow.com/a/1176023
pattern1 = re.compile(r'(.)([A-Z][a-z]+)')
pattern2 = re.compile(r'([a-z0-9])([A-Z])')


def camel_to_snake(name):
    """
    Convert CamelCase to snake_case.

    From https://stackoverflow.com/a/1176023
    """
    name = pattern1.sub(r'\1_\2', name)
    return pattern2.sub(r'\1_\2', name).lower()


def get_metric_name(metric):
    if isinstance(metric, tuple):
        return metric

    if hasattr(metric, '__name__'):
        name = metric.__name__
    elif hasattr(metric, '__class__'):
        name = camel_to_snake(metric.__class__.__name__)
    else:
        name = 'unknown_metric'

    name = name[:-5] if name.endswith('_loss') else name
    return name, metric


def rename_doubles(metric_names):
    counts = Counter(metric_names)
    numbering = Counter()
    ret = []
    for name in metric_names:
        if counts[name] > 1:
            numbering[name] += 1
            ret.append(name + str(numbering[name]))
        else:
            ret.append(name)

    return ret


def get_callables_and_names(metrics):
    if len(metrics) != 0:
        metrics = list(map(get_metric_name, metrics))
        names, metrics = tuple(zip(*metrics))
        names = rename_doubles(names)
        # Make sure that batch_metrics and epoch_metrics are both lists.
        return list(metrics), list(names)
    return [], []
