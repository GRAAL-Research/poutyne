.. role:: hidden
    :class: hidden-section

Metrics
=======

.. currentmodule:: poutyne

Poutyne offers two kind of metrics: batch metrics and epoch metrics.
The main difference between them is that **batch metrics** are computed at each batch and averaged at the end of an epoch whereas **epoch metrics** compute statistics for each batch and compute the metric at the end of the epoch.

Epoch metrics offer a way to compute metrics that are not decomposable as an average.
For instance, the predefined :class:`~poutyne.F1` epoch metric implements the F1 score.
One who knows what an F1-score may know that an average of multiple F1 scores is not the equivalent of the overall F1 score.

Batch Metrics
-------------

Batch metrics are computed at each batch and averaged at the end of an epoch.
The interface is the same as PyTorch loss function, that is ``metric(y_pred, y_true)``.

In addition to the predefined batch metrics below, all PyTorch loss functions can be used by string name in the :class:`batch_metrics argument <poutyne.Model>` under their `functional <https://pytorch.org/docs/stable/nn.functional.html>`_ name.
The key in :class:`callback logs<poutyne.Callback>` associated with each of them is the same as its name but without the ``_loss`` suffix. For example, the loss function :func:`~torch.nn.functional.mse_loss` can be passed as a batch metric with the name ``'mse_loss'`` or simply ``'mse'`` and the keys are going to be ``'mse'`` and ``'val_mse'`` for the training and validation MSE, respectively.
Note that you can also pass the PyTorch loss functions as a loss function in :class:`~poutyne.Model` in the same way.

.. autofunction:: acc
.. autofunction:: bin_acc

Epoch Metrics
-------------

Epoch metrics are metrics calculated **only** at the end of every epoch.
They need to be implemented following the interface class, but we provide an few predefined metrics.

Interface
~~~~~~~~~

.. autoclass:: EpochMetric
    :members:

Predefined Epoch Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FBeta
    :members:
.. autoclass:: F1
.. autoclass:: SKLearnMetrics
    :members:
