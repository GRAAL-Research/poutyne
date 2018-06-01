.. role:: hidden
    :class: hidden-section

pytoune.framework.callbacks
===========================

.. automodule:: pytoune
.. currentmodule:: pytoune.framework.callbacks

Callbacks are a way to interact with the optimization process. For instance, the
``ModelCheckpoint`` callback allows to save the weights of the epoch that has the
best "score", or the ``EarlyStopping`` callback allows to stop the training
when the "score" has not gone up for a while, etc. The following presents the
callbacks available in PyToune, but first the documentation of the ``Callback``
class shows which methods are available in the callback and what arguments they
are provided with.

Callback class
--------------

.. autoclass:: Callback
    :members:

PyToune's Callbacks
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

.. autoclass:: TensorBoardLogger

Checkpointing
-------------

PyToune provides callbacks for checkpointing the state of the optimization
so that it can be stopped and restarted at a later point. All the checkpointing
classes inherit the ``PeriodicSaveCallback`` class and, thus, have the same
arguments in their constructors. They may have other arguments specific to their
purpose.

.. autoclass:: PeriodicSaveCallback

.. autoclass:: ModelCheckpoint

.. autoclass:: OptimizerCheckpoint

.. autoclass:: LRSchedulerCheckpoint

.. autoclass:: PeriodicSaveLambda

LR Schedulers
-------------

PyToune's callbacks for learning rate schedulers are just wrappers around `PyTorch's learning
rate schedulers <http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate>`_
and thus have the same arguments except for the optimizer that has to be
omitted.

.. autoclass:: LambdaLR

.. autoclass:: StepLR

.. autoclass:: MultiStepLR

.. autoclass:: ExponentialLR

.. autoclass:: CosineAnnealingLR

.. autoclass:: ReduceLROnPlateau
