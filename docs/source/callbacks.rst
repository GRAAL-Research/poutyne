.. role:: hidden
    :class: hidden-section

.. _callbacks:

Callbacks
=========

.. currentmodule:: poutyne

Callbacks are a way to interact with the optimization process. For instance, the
:class:`~poutyne.ModelCheckpoint` callback allows to save the weights of the epoch that has the
best "score", or the :class:`~poutyne.EarlyStopping` callback allows to stop the training
when the "score" has not gone up for a while, etc. The following presents the
callbacks available in Poutyne, but first the documentation of the :class:`~poutyne.Callback`
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

.. autoclass:: ProgressionCallback

Tracking
--------

.. autoclass:: TensorBoardGradientTracker

Checkpointing
-------------

Poutyne provides callbacks for checkpointing the state of the optimization
so that it can be stopped and restarted at a later point. All the checkpointing
classes inherit the :class:`~poutyne.PeriodicSaveCallback` class and, thus, have the same
arguments in their constructors. They may have other arguments specific to their
purpose.

.. autoclass:: PeriodicSaveCallback

.. autoclass:: ModelCheckpoint

.. autoclass:: OptimizerCheckpoint

.. autoclass:: LRSchedulerCheckpoint

.. autoclass:: PeriodicSaveLambda

LR Schedulers
-------------

Poutyne's callbacks for learning rate schedulers are just wrappers around `PyTorch's learning rate
schedulers <http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate>`__ and thus have
the same arguments except for the optimizer that has to be omitted.

.. autoclass:: LambdaLR
.. autoclass:: MultiplicativeLR
.. autoclass:: StepLR
.. autoclass:: MultiStepLR
.. autoclass:: ExponentialLR
.. autoclass:: CosineAnnealingLR
.. autoclass:: CyclicLR
.. autoclass:: OneCycleLR
.. autoclass:: CosineAnnealingWarmRestarts
.. autoclass:: ReduceLROnPlateau

Policies
--------

The ``policies`` module is an alternative way to configure your training process. It gives you fine
grained control over the process.

The training is divided into phases with the :class:`~poutyne.Phase` class.
A :class:`~poutyne.Phase` contains parameter spaces (e.g. learning rate,
or momentum, or both) for the optimizer. You chain :class:`~poutyne.Phase`
instances by passing them to the :class:`~poutyne.OptimizerPolicy`.
:class:`~poutyne.OptimizerPolicy` is a :class:`~poutyne.Callback`
that uses the phases, steps through them, and sets the parameters of the optimizer.

.. autoclass:: Phase

.. autoclass:: OptimizerPolicy

.. autofunction:: linspace

.. autofunction:: cosinespace

High Level Policies
~~~~~~~~~~~~~~~~~~~

Ready to use policies.

.. autofunction:: one_cycle_phases

.. autofunction:: sgdr_phases
