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


def _assert_list_length_with_num_metrics(l, metrics, name):
    if l is not None and len(l) != len(metrics):
        raise ValueError(
            f"A {name} was not provided for each metric. " f"Got {len(l)} {name}s for {len(metrics)} metrics."
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
    """[summary]

    Args:
        history (Union[List[Dict[str, Union[float, int]]], pd.DataFrame]): [description]
        metrics (Optional[List[str]], optional): [description]. Defaults to None.
        labels (Optional[List[str]], optional): [description]. Defaults to None.
        titles (Optional[Union[List[str], str]], optional): [description]. Defaults to None.
        axes (Optional[List[matplotlib.axes.Axes]], optional): [description]. Defaults to None.
        show (bool, optional): [description]. Defaults to True.
        save (bool, optional): [description]. Defaults to False.
        save_filename_template (str, optional): [description]. Defaults to '{metric}'.
        save_directory (Optional[str], optional): [description]. Defaults to None.
        save_extensions (Union[List[str], Tuple[str]], optional): [description]. Defaults to `('png', )`.
        close (Optional[bool], optional): [description]. Defaults to None.
        fig_kwargs (Optional[Dict[str, Any]], optional): [description]. Defaults to None.

    Returns:
        Tuple[List[matplotlib.figure.Figure], List[matplotlib.axes.Axes]]: [description]
    """

    # pylint: disable=too-many-locals
    _raise_error_if_matplotlib_not_there()

    metrics = _infer_metrics(history, metrics)

    _assert_list_length_with_num_metrics(labels, metrics, 'label')

    if isinstance(titles, str):
        titles = [titles] * len(metrics)
    else:
        _assert_list_length_with_num_metrics(titles, metrics, 'title')

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
    """[summary]

    Args:
        history (Union[List[Dict[str, Union[float, int]]], pd.DataFrame]): [description]
        metric (str): [description]
        label (str, Optional[str]): [description]. Defaults to None.
        title (str, optional): [description]. Defaults to ''.
        ax (Optional[matplotlib.axes.Axes], optional): [description]. Defaults to None.
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
