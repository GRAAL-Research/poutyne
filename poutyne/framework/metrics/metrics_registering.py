"""
Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.
"""

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


metric_funcs_dict = {}


def clean_metric_func_name(name):
    name = name.lower()
    name = name[:-4] if name.endswith('loss') else name
    name = name.replace('_', '')
    return name


def do_register_metric_func(func, names=None, unique_name=None):
    names = [func.__name__] if names is None or len(names) == 0 else names
    names = [names] if isinstance(names, str) else names
    names = [clean_metric_func_name(name) for name in names]
    if unique_name is None:
        update = {name: func for name in names}
    else:
        update = {name: (unique_name, func) for name in names}
    metric_funcs_dict.update(update)
    return names


register_metric_func = _get_registering_decorator(do_register_metric_func)

metric_classes_dict = {}


def clean_metric_class_name(name):
    name = name.lower()
    name = name[:-5] if name.endswith('score') else name
    name = name.replace('_', '')
    return name


def do_register_metric_class(clz, names=None, unique_name=None):
    names = [camel_to_snake(clz.__name__)] if names is None or len(names) == 0 else names
    names = [names] if isinstance(names, str) else names
    names = [clean_metric_class_name(name) for name in names]
    if unique_name is None:
        update = {name: clz for name in names}
    else:
        update = {name: (unique_name, clz) for name in names}
    metric_classes_dict.update(update)
    return names


def unregister_metric_class(names):
    for name in names:
        del metric_classes_dict[name]


register_metric_class = _get_registering_decorator(do_register_metric_class)


def get_metric(metric):
    if isinstance(metric, str):
        return _get_metric(metric)
    if isinstance(metric, tuple) and isinstance(metric[1], str):
        name, metric = metric
        metric = _get_metric(metric)
        if isinstance(metric, tuple):
            metric = metric[1]
        return name, metric
    return metric


def _get_metric(metric_name):
    loss_metric_func = clean_metric_func_name(metric_name)
    metric_class = clean_metric_class_name(metric_name)

    if loss_metric_func in metric_funcs_dict:
        return metric_funcs_dict[loss_metric_func]

    if metric_class in metric_classes_dict:
        metric = metric_classes_dict[metric_class]
        if isinstance(metric, tuple):
            name, metric = metric
            return name, metric()
        return metric()

    raise ValueError(f"Invalid metric name {repr(metric_name)}.")
