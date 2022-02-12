.. role:: hidden
    :class: hidden-section

Metrics
=======

.. currentmodule:: poutyne

Poutyne offers three kinds of metrics: batch metrics, epoch metrics and `torch metrics <https://torchmetrics.readthedocs.io/>`__.
The main difference between batch and epoch metrics is that **batch metrics** are computed at each batch and averaged at the end of an epoch whereas **epoch metrics** compute statistics for each batch and compute the metric at the end of the epoch.
**Torch metrics** can be passed either as epoch metrics (only displayed at the end of the epoch) or using the specialized :class:`torch_metrics argument <poutyne.Model>` (displayed for each batch and at the end of the epoch).

Epoch metrics offer a way to compute metrics that are not decomposable as an average.
For instance, the predefined :class:`~poutyne.F1` epoch metric implements the F1 score.
One who knows what an F1-score may know that an average of multiple F1 scores is not the equivalent of the overall F1 score.

Here is an example using the three kinds of metrics:

.. code-block:: python

  from poutyne import Model, Accuracy, F1
  import torchmetrics

  model = Model(
      network,
      'sgd',
      'cross_entropy',

      batch_metrics=[Accuracy()],
      # Can also use a string in this case:
      # batch_metrics=['accuracy'],

      epoch_metrics=[F1(), torchmetrics.AUROC(num_classes=10)],
      # Can also use a string in the case of F1-score:
      # epoch_metrics=['f1', torchmetrics.AUROC(num_classes=10)],

      # TorchMetrics can also bu put in the torch_metrics argument
      # to show the metric for each batch but sometimes this significantly
      # slow down the computations.
      torch_metrics=[torchmetrics.AUROC(num_classes=10)],
  )
  model.fit_dataset(train_dataset, valid_dataset)


.. _batch metrics:

Batch Metrics
-------------

Batch metrics are computed at each batch and averaged at the end of an epoch.
The interface is the same as PyTorch loss function, that is ``metric(y_pred, y_true)``.
They are passed to :class:`~poutyne.Model` and :meth:`ModelBundle.from_network() <poutyne.ModelBundle.from_network()>` using the ``batch_metrics`` argument.

In addition to the predefined batch metrics below, all PyTorch loss functions can be used by string name in the :class:`batch_metrics argument <poutyne.Model>` under their `functional <https://pytorch.org/docs/stable/nn.functional.html>`_ name.
The key in :class:`callback logs<poutyne.Callback>` associated with each of them is the same as its name but without the ``_loss`` suffix. For example, the loss function :func:`~torch.nn.functional.mse_loss` can be passed as a batch metric with the name ``'mse_loss'`` or simply ``'mse'`` and the keys are going to be ``'mse'`` and ``'val_mse'`` for the training and validation MSE, respectively.
Note that you can also pass the PyTorch loss functions as a loss function in :class:`~poutyne.Model` in the same way.

Here is an example using batch metrics:

.. code-block:: python

  from poutyne import Model, Accuracy

  model = Model(
      network,
      'sgd',
      'cross_entropy',

      batch_metrics=[Accuracy()],
      # Can also use a string in this case:
      # batch_metrics=['accuracy'],
  )
  model.fit_dataset(train_dataset, valid_dataset)


.. _object oriented batch metrics:

Object-Oriented API
~~~~~~~~~~~~~~~~~~~

Below are classes for predefined batch metrics available in Poutyne.

.. autoclass:: Accuracy
.. autoclass:: BinaryAccuracy
.. autoclass:: TopKAccuracy


Functional
~~~~~~~~~~

Below is the functional version of the classes in the :ref:`object oriented batch metrics` section.

.. autofunction:: acc
.. autofunction:: bin_acc
.. autofunction:: topk


.. _epoch metrics:

Epoch Metrics
-------------

Epoch metrics are metrics calculated **only** at the end of every epoch.
They need to be implemented either inheriting the :class:`~poutyne.EpochMetric` interface or using the :class:`torchmetrics.Metric <torchmetrics.Metric>` interface.
They are passed to :class:`~poutyne.Model` and :meth:`ModelBundle.from_network() <poutyne.ModelBundle.from_network()>` using the ``epoch_metrics`` argument.
In addition to those available in `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/references/modules.html>`__, we also provide a few predefined epoch metrics.

Here is an example using epoch metrics:

.. code-block:: python

  from poutyne import Model, F1
  import torchmetrics

  model = Model(
      network,
      'sgd',
      'cross_entropy',

      epoch_metrics=[F1(), torchmetrics.AUROC(num_classes=10)],
      # Can also use a string in the case of F1-score:
      # epoch_metrics=['f1', torchmetrics.AUROC(num_classes=10)],
  )
  model.fit_dataset(train_dataset, valid_dataset)

Interface
~~~~~~~~~

.. autoclass:: EpochMetric
    :members:

Predefined Epoch Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FBeta
    :members:
.. autoclass:: F1
.. autoclass:: Precision
.. autoclass:: Recall
.. autoclass:: SKLearnMetrics
    :members:


TorchMetrics
------------

`TorchMetrics <https://torchmetrics.readthedocs.io/>`__ is a library implementing a lot of known metrics in PyTorch.
They are passed to :class:`~poutyne.Model` and :meth:`ModelBundle.from_network() <poutyne.ModelBundle.from_network()>` using either the  ``epoch_metrics`` argument or the ``torch_metrics`` argument.
When passed as epoch metric, they are only displayed at the end of the epoch.
When passed using specialized ``torch_metrics`` argument, they are ddisplayed for each batch and at the end of the epoch.
See `TorchMetrics documentation <https://torchmetrics.readthedocs.io/en/latest/references/modules.html>`__ for available torch metrics.

.. warning:: When using this argument, the torch metrics are computed at each batch. This
    can significantly slow down the computations depending on the metrics used. In such case, we advise to
    use them as epoch metrics instead.

Here is an example using TorchMetrics:

.. code-block:: python

  from poutyne import Model
  import torchmetrics

  model = Model(
      network,
      'sgd',
      'cross_entropy',
      torch_metrics=[torchmetrics.F1(num_classes=10)],
  )
  model.fit_dataset(train_dataset, valid_dataset)


.. _multiple metrics at once:

Computing Multiple Metrics at Once
----------------------------------

When passing the metrics to :class:`~poutyne.Model` and :meth:`ModelBundle.from_network() <poutyne.ModelBundle.from_network()>`, the name of each batch and epoch metric can be change by passing a tuple ``(name, metric)`` instead of simply the metric function or object, where ``name`` is the alternative name of the metric.

Batch and epoch metrics can return multiple metrics (e.g. an epoch metric could return an F1-score with the associated precision and recall).
The metrics can be returned via an iterable (tuple, list, Numpy arrays, tensors, etc.) or via a mapping (e.g. a dict).
However, in this case, the names of the different metric has to be passed in some way.

There are two ways to do so.
The easiest one is to pass the metric as a tuple ``(names, metric)`` where ``names`` is a tuple containing a name for each metric returned.
Another way is to override the attribute ``__name__`` of the function or object so that it returns a tuple containing a name for all metrics returned.
Note that, when the metric returns a mapping, the names of the different metrics must be keys in the mapping.

Example with batch metrics:

.. code-block:: python

    # Example with custom batch metrics
    my_custom_metric = lambda input, target: 42.
    my_custom_metric2 = lambda input, target: torch.tensor([42., 43.])
    my_custom_metric3 = lambda input, target: {'a': 42., 'b': 43.}
    batch_metrics = [
        ('custom_name', my_custom_metric),
        (('metric_1', 'metric_2'), my_custom_metric2),
        (('a', 'b'), my_custom_metric3),
    ]

Example with epoch metrics:

.. code-block:: python

    from poutyne import EpochMetric

    class CustomEpochMetric(EpochMetric):
        def forward(self, y_pred, y_true):
            pass

        def get_metric(self):
            return torch.tensor([42., 43.])

        def reset(self):
            pass

    class CustomEpochMetric2(EpochMetric):
        def forward(self, y_pred, y_true):
            pass

        def get_metric(self):
            return {'a': 42., 'b': 43.}

        def reset(self):
            pass


    class CustomEpochMetric3(EpochMetric):
        def __init__(self):
            super().__init__()
            self.__name__ = ['c', 'd']

        def forward(self, y_pred, y_true):
            pass

        def get_metric(self):
            return torch.tensor([42., 43.])

        def reset(self):
            pass

    epoch_metrics = [
        (('metric_1', 'metric_2'), CustomEpochMetric()),
        (('a', 'b'), CustomEpochMetric2()),
        CustomEpochMetric3(),  # No need to pass the names since the class sets the attribute __name__.
    ]

Example with TorchMetrics:

.. code-block:: python

  from torchmetrics import F1, Precision, Recall, MetricCollection

  metric_collection = MetricCollection([
      F1(num_classes=10, average='macro'),
      Precision(num_classes=10, average='macro'),
      Recall(num_classes=10, average='macro')
  ])

  # The names are the keys returned by MetricCollection.
  torch_metrics = [
      (('F1', 'Precision', 'Recall'), metric_collection),
  ]
