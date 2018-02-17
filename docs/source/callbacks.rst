.. role:: hidden
    :class: hidden-section

pytoune.framework.callbacks
==================================================

.. automodule:: pytoune
.. currentmodule:: pytoune.framework.callbacks

.. autoclass:: Callback

Callbacks
---------

.. autoclass:: TerminateOnNaN

.. autoclass:: ModelCheckpoint

.. autoclass:: BestModelRestore

.. autoclass:: EarlyStopping

.. autoclass:: CSVLogger

.. autoclass:: DelayCallback

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
