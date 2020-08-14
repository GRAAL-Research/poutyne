.. Poutyne documentation master file, created by
   sphinx-quickstart on Sat Feb 17 12:19:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/GRAAL-Research/poutyne

.. meta::
  :description: Poutyne is a Keras-like framework for PyTorch and handles much of the boilerplating code needed to train neural networks.
  :keywords: poutyne, poutine, deep learning, pytorch, neural network, keras, machine learning, data science, python
  :author: Frédérik Paradis
  :property="og:image": https://poutyne.org/_static/logos/poutyne-notext.png

Here is Poutyne
===============

Poutyne is a Keras-like framework for `PyTorch <https://pytorch.org/>`_ and handles much of the boilerplating code needed to train neural networks.

Use Poutyne to:

- Train models easily.
- Use callbacks to save your best model, perform early stopping and much more.

Read the documentation at `Poutyne.org <https://poutyne.org>`_.

Poutyne is compatible with the **latest version of PyTorch** and  **Python >= 3.6**.

Cite
----
.. code-block:: bib

  @misc{poutyne,
      author = {Paradis, Fr{\'e}d{\'e}rik},
      title  = {{Poutyne: A Keras-like framework for PyTorch}},
      year   = {2018},
      note   = {\url{https://poutyne.org}}
  }


Getting started: few seconds to Poutyne
=======================================

The core data structure of Poutyne is a :class:`~poutyne.framework.Model`, a way to train your own `PyTorch <https://pytorch.org/docs/master/nn.html>`_ neural networks.

How Poutyne works is that you create your `PyTorch <https://pytorch.org/docs/master/nn.html>`_ module (neural network) as usual but when comes the time to train it you feed it into the Poutyne Model, which handles all the steps, stats and callbacks, similar to what `Keras <https://keras.io/>`_ does.

Here is a simple example:

.. code-block:: python

  # Import the Poutyne Model and define a toy dataset
  from poutyne.framework import Model
  import torch
  import numpy as np

  num_features = 20
  num_classes = 5

  num_train_samples = 800
  train_x = np.random.randn(num_train_samples, num_features).astype('float32')
  train_y = np.random.randint(num_classes, size=num_train_samples).astype('int64')

  num_valid_samples = 200
  valid_x = np.random.randn(num_valid_samples, num_features).astype('float32')
  valid_y = np.random.randint(num_classes, size=num_valid_samples).astype('int64')

  num_test_samples = 200
  test_x = np.random.randn(num_test_samples, num_features).astype('float32')
  test_y = np.random.randint(num_classes, size=num_test_samples).astype('int64')


Create yourself a `PyTorch <https://pytorch.org/docs/master/nn.html>`_ network:

.. code-block:: python

  pytorch_network = torch.nn.Linear(num_features, num_classes)


You can now use Poutyne's model to train your network easily:

.. code-block:: python

  model = Model(pytorch_network, 'sgd', 'cross_entropy',
                batch_metrics=['accuracy'], epoch_metrics=['f1'])
  model.fit(
      train_x, train_y,
      validation_data=(valid_x, valid_y),
      epochs=5,
      batch_size=32
  )


This is really similar to the `model.compile <https://keras.io/models/model/#compile>`_ and `model.fit <https://keras.io/models/model/#fit>`_ functions as in `Keras <https://keras.io/>`_.


You can evaluate the performances of your network using the ``evaluate`` method of Poutyne's model:

.. code-block:: python

  loss_and_metrics = model.evaluate(test_x, test_y)


Or only predict on new data:

.. code-block:: python

  predictions = model.predict(test_x)


As you can see, Poutyne is inspired a lot by the friendliness of `Keras <https://keras.io/>`_. See the Poutyne documentation at `Poutyne.org <https://poutyne.org/>`_ for more.


Installation
============

Before installing Poutyne, you must have the latest version of `PyTorch <https://pytorch.org/>`_ in your environment.

- **Install the stable version of Poutyne:**

  .. code-block:: sh

    pip install poutyne

- **Install the latest development version of Poutyne:**

  .. code-block:: sh

    pip install -U git+https://github.com/GRAAL-Research/poutyne.git@dev


Learning Material
=================

Examples
--------

Look at notebook files with full working `examples <https://github.com/GRAAL-Research/poutyne/blob/master/examples/>`_:

- `introduction_pytorch_poutyne.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/introduction_pytorch_poutyne.ipynb>`_  (`tutorial version <https://github.com/GRAAL-Research/poutyne/blob/master/tutorials/introduction_pytorch_poutyne_tutorial.ipynb>`_) - comparison of Poutyne with bare PyTorch and example of a Poutyne callback.
- `transfer_learning.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/transfer_learning.ipynb>`_ - transfer learning on ``ResNet-18`` on the `CUB-200 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ dataset.
- `policy_cifar_example.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/policy_cifar_example.ipynb>`_ - policies API, FastAI-like learning rate policies
- `policy_interface.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/policy_interface.ipynb>`_ - example of policies
- `tips_and_tricks.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/tips_and_tricks.ipynb>`_ -  tips and tricks using Poutyne.

Videos
------

* `Presentation on Poutyne <https://youtu.be/gQ3SW5r7HSs>`_ given at one of the weekly presentations of the Institute Intelligence and Data (IID) of Université Laval. `Slides <https://github.com/GRAAL-Research/poutyne/blob/master/slides/poutyne.pdf>`_ and the `associated Latex source code <https://github.com/GRAAL-Research/poutyne/blob/master/slides/src/>`_ are also available.

Contributing to Poutyne
=======================

We welcome user input, whether it is regarding bugs found in the library or feature propositions ! Make sure to have a look at our `contributing guidelines <https://github.com/GRAAL-Research/poutyne/blob/master/CONTRIBUTING.md>`_ for more details on this matter.


License
=======

Poutyne is LGPLv3 licensed, as found in the `LICENSE file <https://github.com/GRAAL-Research/poutyne/blob/master/LICENSE>`_.


Why this name, Poutyne?
=======================

Poutyne (or poutine in Québécois) is now the well-known dish from Quebec composed of French fries, squeaky cheese curds and brown gravy. However, in Quebec, it also has the meaning of something that is an `"ordinary or common subject or activity" <https://fr.wiktionary.org/wiki/poutine>`_. Thus, Poutyne will get rid of the ordinary boilerplate code that plain `PyTorch <https://pytorch.org/>`_ training usually entails.

.. figure:: https://upload.wikimedia.org/wikipedia/commons/4/4e/La_Banquise_Poutine_%28cropped%29.jpg
  :alt: Poutine

  Yuri Long from Arlington, VA, USA [`CC BY 2.0 <https://creativecommons.org/licenses/by/2.0>`_]


API Reference
=============

.. toctree::
  :maxdepth: 1
  :caption: API

  model
  experiment
  metrics
  callbacks
  layers
  utils

.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Examples

  examples/introduction
  examples/tips_and_tricks
  examples/policy_interface
  examples/train_with_policy_module
  examples/transfert_learning


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
