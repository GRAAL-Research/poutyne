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


def get_names_of_metric(metric):
    if isinstance(metric, tuple):
        names, metric = metric
    elif hasattr(metric, '__name__'):
        names = metric.__name__
    elif hasattr(metric, '__class__'):
        names = camel_to_snake(metric.__class__.__name__)
    else:
        names = 'unknown_metric'
    return names, metric


def flatten_metric_names(metric_names):
    to_list = lambda names: names if isinstance(names, list) else [names]
    return [name for names in metric_names for name in to_list(names)]


def rename_doubles(batch_metrics_names, epoch_metrics_names):
    counts = Counter(flatten_metric_names(batch_metrics_names + epoch_metrics_names))
    numbering = Counter()
    batch_metrics_names = rename_doubles_from_counts(batch_metrics_names, counts, numbering)
    epoch_metrics_names = rename_doubles_from_counts(epoch_metrics_names, counts, numbering)
    return batch_metrics_names, epoch_metrics_names


def rename_doubles_from_counts(metric_names, counts, numbering):
    """
    This function takes a list in the format `['a', ['b', 'a'], 'c', 'a', 'c']`
    and returns a list where each double is added a number so that there are no
    more doubles in the list: `['a1', ['b', 'a2'], 'c1', 'a3', 'c2']`. It does so
    using the provided counts and using the numbering Counter object.
    """

    def get_name(name):
        if counts[name] > 1:
            numbering[name] += 1
            return name + str(numbering[name])
        return name

    return [[get_name(name) for name in names] if not isinstance(names, str) else get_name(names)
            for names in metric_names]


def get_callables_and_names(metrics):
    if len(metrics) != 0:
        metrics = list(map(get_names_of_metric, metrics))
        names, metrics = tuple(zip(*metrics))
        # Make sure that batch_metrics and epoch_metrics are both lists.
        return list(metrics), list(names)
    return [], []
