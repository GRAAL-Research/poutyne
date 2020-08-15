.. role:: hidden
    :class: hidden-section

Callbacks
=========

.. currentmodule:: poutyne.framework.callbacks

Callbacks are a way to interact with the optimization process. For instance, the
:class:`~poutyne.framework.callbacks.ModelCheckpoint` callback allows to save the weights of the epoch that has the
best "score", or the :class:`~poutyne.framework.callbacks.EarlyStopping` callback allows to stop the training
when the "score" has not gone up for a while, etc. The following presents the
callbacks available in Poutyne, but first the documentation of the :class:`~poutyne.framework.callbacks.Callback`
class shows which methods are available in the callback and what arguments they
are provided with.

Callback class
--------------

.. autoclass:: Callback
    :members:

Poutyne's Callbacks
-------------------

.. autoclass:: TerminateOnNaN

.. autoclass:: BestModelRestore

.. autoclass:: EarlyStopping

.. autoclass:: DelayCallback

.. autoclass:: ClipNorm

.. autoclass:: ClipValue

Logging
-------

.. autoclass:: CSVLogger

.. autoclass:: AtomicCSVLogger

.. autoclass:: TensorBoardLogger

Tracking
--------

.. autoclass:: TensorBoardGradientTracker

Checkpointing
-------------

Poutyne provides callbacks for checkpointing the state of the optimization
so that it can be stopped and restarted at a later point. All the checkpointing
classes inherit the :class:`~poutyne.framework.callbacks.PeriodicSaveCallback` class and, thus, have the same
arguments in their constructors. They may have other arguments specific to their
purpose.

.. autoclass:: PeriodicSaveCallback

.. autoclass:: ModelCheckpoint

.. autoclass:: OptimizerCheckpoint

.. autoclass:: LRSchedulerCheckpoint

.. autoclass:: PeriodicSaveLambda

LR Schedulers
-------------

.. automodule:: poutyne.framework.callbacks.lr_scheduler
    :members:
    :exclude-members: ReduceLROnPlateau

.. autoclass:: ReduceLROnPlateau

Policies
--------

.. automodule:: poutyne.framework.callbacks.policies

.. autoclass:: Phase

.. autoclass:: OptimizerPolicy

.. autofunction:: linspace

.. autofunction:: cosinespace

High Level Policies
~~~~~~~~~~~~~~~~~~~

Ready to use policies.

.. autofunction:: one_cycle_phases

.. autofunction:: sgdr_phases
