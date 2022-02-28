.. role:: hidden
    :class: hidden-section

.. _metrics:

Metrics
=======

.. currentmodule:: poutyne

Poutyne offers two kinds of metrics: batch and epoch metrics.
The main difference between batch and epoch metrics is that **batch metrics** are computed at each batch, whereas **epoch metrics** compute statistics for each batch and compute the metric at the end of the epoch.
Batch metrics are passed to :class:`~poutyne.Model` and :meth:`ModelBundle.from_network() <poutyne.ModelBundle.from_network()>` using the ``batch_metrics`` argument.
Epoch metrics are passed to :class:`~poutyne.Model` and :meth:`ModelBundle.from_network() <poutyne.ModelBundle.from_network()>` using the ``epoch_metrics`` argument.

In addition to the predefined metrics below, all PyTorch loss functions can be used by string name under their `functional <https://pytorch.org/docs/stable/nn.functional.html#loss-functions>`_ name.
The key in :class:`callback logs<poutyne.Callback>` associated with each is the same as its name but without the ``_loss`` suffix. For example, the loss function :func:`~torch.nn.functional.mse_loss` can be passed as a metric with the name ``'mse_loss'`` or simply ``'mse'``, and the keys will be ``'mse'`` and ``'val_mse'`` for the training and validation MSE, respectively.
Note that you can also pass the PyTorch loss functions as a loss function in :class:`~poutyne.Model` in the same way.

.. warning:: When using the ``batch_metrics`` argument, the metrics are computed for each batch.
    This can significantly slow down the computations depending on the metrics used.
    This mostly happens on non-decomposable metrics such as :class:`torchmetrics.AUROC <torchmetrics.AUROC>` where an ordering of the elements is necessary to compute the metric.
    In such a case, we advise using them as epoch metrics instead.

Here is an example using metrics:

.. code-block:: python

    from poutyne import Model, Accuracy, F1
    import torchmetrics

    model = Model(
        network,
        'sgd',
        'cross_entropy',

        batch_metrics=[Accuracy(), F1()],
        # Can also use a string in this case:
        # batch_metrics=['accuracy', 'f1'],

        epoch_metrics=[torchmetrics.AUROC(num_classes=10)],
    )
    model.fit_dataset(train_dataset, valid_dataset)


Interface
---------

There are two interfaces available for metrics.
The first interface is the same as PyTorch loss functions: ``metric(y_pred, y_true)``.
When using that interface, the metric is assumed to be decomposable and is average for the whole epoch.
The batch size is inferred with :func:`poutyne.get_batch_size()` using `y_pred` and `y_true` as values.

The second interface is defined by the :class:`~poutyne.Metric` class.
As documented in the class, it provides methods for updating and computing the metric.
This interface is compatible with `TorchMetrics <https://torchmetrics.readthedocs.io/>`__, a library implementing many known metrics in PyTorch.
See the `TorchMetrics documentation <https://torchmetrics.readthedocs.io/en/latest/references/modules.html>`__ for available TorchMetrics metrics.

Note that if one implements a metric intended as both a batch and epoch metric, the methods :meth:`Metric.forward()` and :meth:`Metric.update()` need to be implemented.
To avoid implementing both methods, one can `implement a TorchMetrics metric <https://torchmetrics.readthedocs.io/en/latest/pages/implement.html>`__ at the potential cost of higher computational load as described in the `TorchMetrics documentation <https://torchmetrics.readthedocs.io/en/latest/pages/implement.html#internal-implementation-details>`__.


.. autoclass:: Metric
    :members:

.. _object oriented metrics:

Object-Oriented API
-------------------

Below are classes for predefined metrics available in Poutyne.

.. autoclass:: Accuracy
.. autoclass:: BinaryAccuracy
.. autoclass:: TopKAccuracy
.. autoclass:: FBeta
    :members:
.. autoclass:: F1
.. autoclass:: Precision
.. autoclass:: Recall
.. autoclass:: BinaryF1
.. autoclass:: BinaryPrecision
.. autoclass:: BinaryRecall
.. autoclass:: SKLearnMetrics
    :members:


Functional
----------

Below is a functional version of some of the classes in the :ref:`object oriented metrics` section.

.. autofunction:: acc
.. autofunction:: bin_acc
.. autofunction:: topk


.. _multiple metrics at once:

Computing Multiple Metrics at Once
----------------------------------

When passing the metrics to :class:`~poutyne.Model` and :meth:`ModelBundle.from_network() <poutyne.ModelBundle.from_network()>`, the name of each metric can be change by passing a tuple ``(name, metric)`` instead of simply the metric function or object, where ``name`` is the alternative name of the metric.

Metrics can return multiple metrics (e.g. an metric could return an F1-score with the associated precision and recall).
The metrics can be returned via an iterable (tuple, list, Numpy arrays, tensors, etc.) or via a mapping (e.g. a dict).
However, in this case, the names of the different metric has to be passed in some way.

There are two ways to do so.
The easiest one is to pass the metric as a tuple ``(names, metric)`` where ``names`` is a tuple containing a name for each metric returned.
Another way is to override the attribute ``__name__`` of the function or object so that it returns a tuple containing a name for all metrics returned.
Note that, when the metric returns a mapping, the names of the different metrics must be keys in the mapping.

Examples:

.. code-block:: python

    import torch
    from poutyne import Metric
    from torchmetrics import F1Score, Precision, Recall, MetricCollection


    my_custom_metric = lambda input, target: 42.0
    my_custom_metric2 = lambda input, target: torch.tensor([42.0, 43.0])
    my_custom_metric3 = lambda input, target: {'a': 42.0, 'b': 43.0}


    class CustomMetric(Metric):
        def forward(self, y_pred, y_true):
            return self.compute()

        def update(self, y_pred, y_true):
            pass

        def compute(self):
            return torch.tensor([42.0, 43.0])

        def reset(self):
            pass


    class CustomMetric2(Metric):
        def forward(self, y_pred, y_true):
            return self.compute()

        def update(self, y_pred, y_true):
            pass

        def compute(self):
            return {'c': 42.0, 'd': 43.0}

        def reset(self):
            pass


    class CustomMetric3(Metric):
        def __init__(self):
            super().__init__()
            self.__name__ = ['e', 'f']

        def forward(self, y_pred, y_true):
            return self.compute()

        def update(self, y_pred, y_true):
            pass

        def compute(self):
            return torch.tensor([42.0, 43.0])

        def reset(self):
            pass


    metric_collection = MetricCollection(
        [
            F1Score(num_classes=10, average='macro'),
            Precision(num_classes=10, average='macro'),
            Recall(num_classes=10, average='macro'),
        ]
    )

    metrics = [
        ('custom_name', my_custom_metric),
        (('metric_1', 'metric_2'), my_custom_metric2),
        (('a', 'b'), my_custom_metric3),
        (('metric_3', 'metric_4'), CustomMetric()),
        (('c', 'd'), CustomMetric2()),

        # No need to pass the names since the class sets the attribute __name__.
        CustomMetric3(),

        # The names are the keys returned by MetricCollection.
        (('F1Score', 'Precision', 'Recall'), metric_collection),
    ]
