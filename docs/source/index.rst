.. PyToune documentation master file, created by
   sphinx-quickstart on Sat Feb 17 12:19:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/GRAAL-Research/pytoune

.. meta::
   :description: PyToune is a Keras-like framework for PyTorch and handles much of the boilerplating code needed to train neural networks.
   :keywords: deep learning, pytorch, neural network, keras, machine learning, data science, python
   :author: Frédérik Paradis

Here is PyToune
===============

PyToune is a Keras-like framework for `PyTorch <https://pytorch.org/>`_ and handles much of the boilerplating code needed to train neural networks.

Use PyToune to:

- Train models easily.
- Use callbacks to save your best model, perform early stopping and much more.

Read the documentation at `PyToune.org <https://pytoune.org>`_.

PyToune is compatible with **PyTorch >= 0.4.0** and  **Python >= 3.5**.



Getting started: few seconds to PyToune
=======================================

The core data structure of PyToune is a ``Model``, a way to train your own `PyTorch <https://pytorch.org/docs/master/nn.html>`_ neural networks.

How PyToune works is that you create your `PyTorch <https://pytorch.org/docs/master/nn.html>`_ module (neural network) as usual but when comes the time to train it you feed it into the PyToune Model, which handles all the steps, stats and callbacks, similar to what `Keras <https://keras.io/>`_ does.

Here is a simple example:

.. code-block:: python

  # Import the PyToune Model and define a toy dataset
  from pytoune.framework import Model
  import torch
  import numpy as np

  num_features = 20
  num_classes = 5

  num_train_samples = 800
  train_x = np.random.randn(num_train_samples, num_features).astype('float32')
  train_y = np.random.randint(num_classes, size=num_train_samples)

  num_valid_samples = 200
  valid_x = np.random.randn(num_valid_samples, num_features).astype('float32')
  valid_y = np.random.randint(num_classes, size=num_valid_samples)

  num_test_samples = 200
  test_x = np.random.randn(num_test_samples, num_features).astype('float32')
  test_y = np.random.randint(num_classes, size=num_test_samples)


Create yourself a `PyTorch <https://pytorch.org/docs/master/nn.html>`_ network;

.. code-block:: python

  pytorch_module = torch.nn.Linear(num_features, 1)


You can now use PyToune's model to train your network easily;

.. code-block:: python

  model = Model(pytorch_module, 'sgd', 'cross_entropy', metrics=['accuracy'])
  model.fit(
      train_x, train_y,
      validation_x=valid_x,
      validation_y=valid_y,
      epochs=5,
      batch_size=32
    )


This is really similar to the ``model.compile`` function as in `Keras <https://keras.io/>`_;

.. code-block:: python

  # Keras way to compile and train
  model.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
  model.fit(train_x, train_y, epochs=5, batch_size=32)


You can evaluate the performances of your network using the ``evaluate`` method of PyToune's model;

.. code-block:: python

  loss_and_metrics = model.evaluate(test_x, test_y)


Or only predict on new data;

.. code-block:: python

  predictions = model.predict(test_x)


As you can see, PyToune is inspired a lot by the friendliness of `Keras <https://keras.io/>`_. See the PyToune documentation at `PyToune.org <https://pytoune.org/>`_ for more.

Installation
============

Before installing PyToune, you must have a working version of `PyTorch 0.4.0 <https://pytorch.org/>`_ in your environment.

- **Install the stable version of PyToune:**

  .. code-block:: sh

    pip install pytoune

- **Install the latest version of PyToune:**

  .. code-block:: sh

    pip install -U git+https://github.com/GRAAL-Research/pytoune.git



Why this name, PyToune?
=======================

PyToune (or pitoune in Québécois) used to be wood logs that flowed through the rivers. It was an efficient way to travel large pieces of wood across the country. We hope that PyToune will make your `PyTorch <https://pytorch.org/>`_ neural networks training flow easily just like the "pitounes" used to.

.. image:: /_static/img/pitounes.jpg
   :alt: Pitounes


API Reference
=============

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   utils
   framework
   callbacks
   layers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
