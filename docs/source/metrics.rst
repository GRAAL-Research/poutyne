.. role:: hidden
    :class: hidden-section

Metrics
=======

.. currentmodule:: poutyne.framework.metrics

Epoch metrics are metrics calculated **only** at the end of every epoch. They need to be implemented following the interface class, but we provide an exhaustive list.

Epoch Metric Interface
----------------------

.. autoclass:: EpochMetric
    :members:

Epoch Metrics
-------------

.. autoclass:: FBeta
    :members:
.. autoclass:: F1
.. autoclass:: SKLearnMetrics
    :members:
