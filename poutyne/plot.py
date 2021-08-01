import os
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    import pandas as pd
except ImportError:
    pd = None

from poutyne import is_in_jupter_notebook

jupyter = is_in_jupter_notebook()


def _none_to_iterator(value, repeat=None):
    return value if value is not None else itertools.repeat(repeat)


def _assert_list_length_with_num_metrics(l, metrics, name, plural_name):
    if l is not None and len(l) != len(metrics):
        raise ValueError(f"A {name} was not provided for each metric. "
                         f"Got {len(l)} {plural_name} for {len(metrics)} metrics.")


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


def _show_and_save_figs(figs, metrics, *, show, save, save_filename_template, save_directory, save_extensions, close):
    if save:
        save_template = save_filename_template
        if save_directory is not None:
            os.makedirs(save_directory, exist_ok=True)
            save_template = os.path.join(save_directory, save_filename_template)

    if close is None and jupyter:
        close = False

    for fig, metric in zip(figs, metrics):
        if show:
            fig.show()

        if save:
            for ext in save_extensions:
                filename = save_template.format(metric=metric) + f'.{ext}'
                fig.savefig(filename)

        if close:
            plt.close(fig)


def plot_history(history,
                 *,
                 metrics=None,
                 labels=None,
                 val_labels=None,
                 titles=None,
                 axes=None,
                 show=True,
                 save=False,
                 save_filename_template='{metric}',
                 save_directory=None,
                 save_extensions=('png', ),
                 close=None,
                 fig_kwargs=None):
    # pylint: disable=too-many-locals
    metrics = _infer_metrics(history, metrics)

    _assert_list_length_with_num_metrics(labels, metrics, 'label', 'labels')
    _assert_list_length_with_num_metrics(val_labels, metrics, 'validation label', 'validation labels')

    if isinstance(titles, str):
        titles = [titles] * len(metrics)
    else:
        _assert_list_length_with_num_metrics(titles, metrics, 'title', 'titles')

    labels = _none_to_iterator(labels)
    val_labels = _none_to_iterator(val_labels)
    titles = _none_to_iterator(titles, repeat='')

    figs, axes = _get_figs_and_axes(axes, len(metrics), fig_kwargs)
    for metric, label, val_label, title, ax in zip(metrics, labels, val_labels, titles, axes):
        plot_metric(history, metric, label=label, val_label=val_label, title=title, ax=ax)

    _show_and_save_figs(figs,
                        metrics,
                        show=show,
                        save=save,
                        save_filename_template=save_filename_template,
                        save_directory=save_directory,
                        save_extensions=save_extensions,
                        close=close)

    return figs, axes


def plot_metric(history, metric, *, label=None, val_label=None, title='', ax=None):
    if ax is None:
        ax = plt.gca()

    if val_label is None:
        if label is not None:
            val_label = 'Validation ' + label
        else:
            val_label = 'val_' + metric

    if label is None:
        label = metric

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

    ax.plot(epochs, metric_values, label=label)
    if val_metric_values is not None:
        ax.plot(epochs, val_metric_values, label=val_label)

    ax.set_xlabel('Epochs')
    ax.set_ylabel(label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend()
    ax.set_title(title)
