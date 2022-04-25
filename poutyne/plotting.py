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

import os
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    matplotlib = True
except ImportError:
    matplotlib = False

try:
    import pandas as pd
except ImportError:
    pd = None

from poutyne import is_in_jupyter_notebook

jupyter = is_in_jupyter_notebook()


def _raise_error_if_matplotlib_not_there():
    if not matplotlib:
        raise ImportError("matplotlib needs to be installed to use this function.")


def _none_to_iterator(value, repeat=None):
    return value if value is not None else itertools.repeat(repeat)


def _assert_list_length_with_num_metrics(list_, metrics, name):
    if list_ is not None and len(list_) != len(metrics):
        raise ValueError(
            f"A {name} was not provided for each metric. Got {len(list_)} {name}s for {len(metrics)} metrics."
        )


def _infer_metrics(history, metrics):
    if metrics is None:
        if pd is not None and isinstance(history, pd.DataFrame):
            cols = list(history.columns)
        else:
            cols = list(history[0].keys())
        metrics = [col for col in cols if col != 'epoch' and not col.startswith('val_')]
    return metrics


def _get_figs_and_axes(axes, num_axes, fig_kwargs):
    figs = ()
    if axes is None:
        fig_kwargs = fig_kwargs if fig_kwargs is not None else {}
        figs, axes = zip(*(plt.subplots(**fig_kwargs) for _ in range(num_axes)))
    return figs, axes


def _save_figs(figs, metrics, *, filename_template, directory, extensions):
    save_template = filename_template
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
        save_template = os.path.join(directory, filename_template)

    for fig, metric in zip(figs, metrics):
        for ext in extensions:
            filename = save_template.format(metric=metric) + f'.{ext}'
            fig.savefig(filename)


def _show_figs(figs):
    for fig in figs:
        fig.show()


def _close_figs(figs):
    for fig in figs:
        plt.close(fig)


def plot_history(
    history: Union[List[Dict[str, Union[float, int]]], 'pd.DataFrame'],
    *,
    metrics: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    titles: Optional[Union[List[str], str]] = None,
    axes: Optional[List['matplotlib.axes.Axes']] = None,
    show: bool = True,
    save: bool = False,
    save_filename_template: str = '{metric}',
    save_directory: Optional[str] = None,
    save_extensions: Union[List[str], Tuple[str]] = ('png',),
    close: Optional[bool] = None,
    fig_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Plot the training history in matplotlib. By default, all metrics are plotted.

    Args:
        history (Union[List[Dict[str, Union[float, int]]], pandas.DataFrame]): The training history to plot. Can be
            either a list of dictionary as returned by :func:`~poutyne.Model.fit()` or a Pandas DataFrame as read from a
            CSV output by the :class:`~poutyne.CSVLogger` callback.
        metrics (Optional[List[str]], optional): The list of metrics for which to output the plot. By default, every
            metric in the history is used.
        labels (Optional[List[str]], optional): A list of labels to use for each metric. Must be of the same length as
            ``metrics``. By default, the names in the history are used.
        titles (Optional[Union[List[str], str]], optional): A title or a list of titles to use for each metric. If a
            list, must be of the same length as ``metrics``. If a string, the same title will be used for all plots. By
            default, there is no title.
        axes (Optional[List[matplotlib.axes.Axes]], optional): A list of matplotlib :class:`~matplotlib.axes.Axes` to
            use for each metric. Must be of the same length as ``metrics``. By default, a new figure and an new axe is
            created for each plot.
        show (bool, optional): Whether to show the plots. Defaults to True.
        save (bool, optional): Whether to save the plots. Defaults to False.
        save_filename_template (str, optional): The filename without extension for saving the plot. Should contain
            ``{metric}`` somewhere in it or all the plots will overwrite each other. Defaults to ``'{metric}'``.
        save_directory (Optional[str], optional): The directory to save the plots. Default to the current directory.
        save_extensions (Union[List[str], Tuple[str]], optional): A list of extensions under which to save the plots.
            Defaults to `('png', )`.
        close (Optional[bool], optional): Whether to close the matplotlib figures. By default, the figures are closed
            except when in Jupyter notebooks.
        fig_kwargs (Optional[Dict[str, Any]], optional): Any keyword arguments to pass to
            :func:`~matplotlib.pyplot.subplots`.

    Returns:
        Tuple[List[matplotlib.figure.Figure], List[matplotlib.axes.Axes]]: A tuple ``(figs, axes)``  where ``figs`` is
        the list of instanciated matplotlib :class:`~matplotlib.figure.Figure` and ``axes`` is a list of instanciated
        matplotlib :class:`~matplotlib.figure.Axes`.
    """

    # pylint: disable=too-many-locals
    _raise_error_if_matplotlib_not_there()

    metrics = _infer_metrics(history, metrics)

    _assert_list_length_with_num_metrics(labels, metrics, 'label')

    if isinstance(titles, str):
        titles = [titles] * len(metrics)
    else:
        _assert_list_length_with_num_metrics(titles, metrics, 'title')

    _assert_list_length_with_num_metrics(axes, metrics, 'axe')

    labels = _none_to_iterator(labels)
    titles = _none_to_iterator(titles, repeat='')

    figs, axes = _get_figs_and_axes(axes, len(metrics), fig_kwargs)
    for metric, label, title, ax in zip(metrics, labels, titles, axes):
        plot_metric(history, metric, label=label, title=title, ax=ax)

    if save:
        _save_figs(
            figs,
            metrics,
            filename_template=save_filename_template,
            directory=save_directory,
            extensions=save_extensions,
        )

    if show:
        _show_figs(figs)

    if close is None:
        close = not jupyter

    if close:
        _close_figs(figs)

    return figs, axes


def plot_metric(
    history: Union[List[Dict[str, Union[float, int]]], 'pd.DataFrame'],
    metric: str,
    *,
    label: Optional[str] = None,
    title: str = '',
    ax: Optional['matplotlib.axes.Axes'] = None,
):
    """
    Plot the training history in matplotlib for a given metric.

    Args:
        history (Union[List[Dict[str, Union[float, int]]], pd.DataFrame]): The training history to plot. Can be
            either a list of dictionary as returned by :func:`~poutyne.Model.fit()` or a Pandas DataFrame as read from a
            CSV output by the :class:`~poutyne.CSVLogger` callback.
        metric (str): The metric for which to output the plot.
        label (str, Optional[str]): A label for the metric. By default, the label is the same as the name of the metric.
        title (str, optional): A title for the plot. By default, no title.
        ax (Optional[matplotlib.axes.Axes], optional): A matplotlib :class:`~matplotlib.axes.Axes` to use. By default,
            the current axe is used.
    """
    _raise_error_if_matplotlib_not_there()

    if ax is None:
        ax = plt.gca()

    if label is None:
        train_label = metric
        valid_label = 'val_' + metric
    else:
        train_label = 'Training ' + label
        valid_label = 'Validation ' + label

    val_metric_values = None
    if pd is not None and isinstance(history, pd.DataFrame):
        epochs = history['epoch']
        metric_values = history[metric]
        if 'val_' + metric in history:
            val_metric_values = history['val_' + metric]
    else:
        epochs = [entry['epoch'] for entry in history]
        metric_values = [entry[metric] for entry in history]
        if 'val_' + metric in history[0]:
            val_metric_values = [entry['val_' + metric] for entry in history]

    ax.plot(epochs, metric_values, label=train_label)
    if val_metric_values is not None:
        ax.plot(epochs, val_metric_values, label=valid_label)

    ax.set_xlabel('Epochs')
    ax.set_ylabel(label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend()
    ax.set_title(title)
