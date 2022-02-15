.. Poutyne documentation master file, created by
   sphinx-quickstart on Sat Feb 17 12:19:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/GRAAL-Research/poutyne

.. meta::
  :description: Poutyne is a simplified framework for PyTorch and handles much of the boilerplating code needed to train neural networks.
  :keywords: poutyne, poutine, deep learning, pytorch, neural network, keras, machine learning, data science, python
  :author: Frédérik Paradis
  :property="og:image": https://poutyne.org/_static/logos/poutyne-notext.png

Here is Poutyne
===============

Poutyne is a simplified framework for `PyTorch <https://pytorch.org/>`_ and handles much of the boilerplating code needed to train neural networks.

Use Poutyne to:

- Train models easily.
- Use callbacks to save your best model, perform early stopping and much more.

Poutyne is compatible with the **latest version of PyTorch** and  **Python >= 3.6**.

Cite
----
.. code-block:: bib

  @misc{Paradis_Poutyne_A_Simplified_2020,
      author = {Paradis, Frédérik and Beauchemin, David and Godbout, Mathieu and Alain, Mathieu and Garneau, Nicolas and Otte, Stefan and Tremblay, Alexis and Bélanger, Marc-Antoine and Laviolette, François},
      title  = {{Poutyne: A Simplified Framework for Deep Learning}},
      year   = {2020},
      url    = {https://poutyne.org}
  }

Getting started: few seconds to Poutyne
=======================================

The core data structure of Poutyne is a :class:`~poutyne.Model`, a way to train your own `PyTorch <https://pytorch.org/docs/master/nn.html>`__ neural networks.

How Poutyne works is that you create your `PyTorch <https://pytorch.org/docs/master/nn.html>`__ module (neural network) as usual but when comes the time to train it you feed it into the Poutyne Model, which handles all the steps, stats and callbacks, similar to what `Keras <https://keras.io/>`_ does.

Here is a simple example:

.. code-block:: python

  # Import the Poutyne Model and define a toy dataset
  from poutyne import Model
  import torch
  import torch.nn as nn
  import numpy as np
  import torchmetrics

  num_features = 20
  num_classes = 5
  hidden_state_size = 100

  num_train_samples = 800
  train_x = np.random.randn(num_train_samples, num_features).astype('float32')
  train_y = np.random.randint(num_classes, size=num_train_samples).astype('int64')

  num_valid_samples = 200
  valid_x = np.random.randn(num_valid_samples, num_features).astype('float32')
  valid_y = np.random.randint(num_classes, size=num_valid_samples).astype('int64')

  num_test_samples = 200
  test_x = np.random.randn(num_test_samples, num_features).astype('float32')
  test_y = np.random.randint(num_classes, size=num_test_samples).astype('int64')


Select a PyTorch device so that it runs on GPU if you have one:

.. code-block:: python

  cuda_device = 0
  device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")


Create yourself a `PyTorch <https://pytorch.org/docs/master/nn.html>`__ network:

.. code-block:: python

  network = nn.Sequential(
      nn.Linear(num_features, hidden_state_size),
      nn.ReLU(),
      nn.Linear(hidden_state_size, num_classes)
  )


You can now use Poutyne's model to train your network easily:

.. code-block:: python

  model = Model(
      network,
      'sgd',
      'cross_entropy',
      batch_metrics=['accuracy'],
      epoch_metrics=['f1', torchmetrics.AUROC(num_classes=num_classes)],
      device=device
  )
  model.fit(
      train_x, train_y,
      validation_data=(valid_x, valid_y),
      epochs=5,
      batch_size=32
  )


Since Poutyne is inspired by `Keras <https://keras.io/>`_, one might have notice that this is really similar to some of its `functions <https://keras.io/models/model/>`_.

You can evaluate the performances of your network using the ``evaluate`` method of Poutyne's model:

.. code-block:: python

  loss, (accuracy, f1score) = model.evaluate(test_x, test_y)


Or only predict on new data:

.. code-block:: python

  predictions = model.predict(test_x)

`See the complete code here. <https://github.com/GRAAL-Research/poutyne/blob/master/examples/basic_random_classification.py>`__ Also, `see this <https://github.com/GRAAL-Research/poutyne/blob/master/examples/basic_random_regression.py>`__ for an example for regression that also uses :ref:`epoch metrics <epoch metrics>`.


One of the strengths Poutyne are :ref:`callbacks <callbacks>`. They allow you to save checkpoints, log training statistics and more. See this `notebook <https://github.com/GRAAL-Research/poutyne/blob/master/examples/introduction_pytorch_poutyne.ipynb>`__ for an introduction to callbacks. In that vein, Poutyne also offers an :class:`~poutyne.ModelBundle` class that offers automatic checkpointing, logging and more using callbacks under the hood. Here is an example of usage.

.. code-block:: python

  from poutyne import ModelBundle

  # Everything is saved in ./saves/my_classification_network
  model_bundle = ModelBundle.from_network(
      './saves/my_classification_network', network, optimizer='sgd', task='classif', device=device
  )

  model_bundle.train_data(train_x, train_y, validation_data=(valid_x, valid_y), epochs=5)

  model_bundle.test_data(test_x, test_y)

`See the complete code here. <https://github.com/GRAAL-Research/poutyne/blob/master/examples/basic_random_classification_with_model_bundle.py>`__ Also, `see this <https://github.com/GRAAL-Research/poutyne/blob/master/examples/basic_random_regression_with_model_bundle.py>`__ for an example for regression that again also uses :ref:`epoch metrics <epoch metrics>`.


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

Blog posts
----------

* `Medium PyTorch post <https://medium.com/pytorch/poutyne-a-simplified-framework-for-deep-learning-in-pytorch-74b1fc1d5a8b>`__ - Presentation of the basics of Poutyne and how it can help you be more efficient when developing neural networks with PyTorch.

Examples
--------

Look at notebook files with full working `examples <https://github.com/GRAAL-Research/poutyne/blob/master/examples/>`_:

- `introduction_pytorch_poutyne.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/introduction_pytorch_poutyne.ipynb>`__  (`tutorial version <https://github.com/GRAAL-Research/poutyne/blob/master/tutorials/introduction_pytorch_poutyne_tutorial.ipynb>`_) - comparison of Poutyne with bare PyTorch and usage examples of Poutyne callbacks and the ModelBundle class.
- `tips_and_tricks.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/tips_and_tricks.ipynb>`__ -  tips and tricks using Poutyne
- `transfer_learning.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/transfer_learning.ipynb>`__ - transfer learning on ``ResNet-18`` on the `CUB-200 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`__ dataset.
- `policy_cifar_example.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/policy_cifar_example.ipynb>`__ - policies API, FastAI-like learning rate policies
- `policy_interface.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/policy_interface.ipynb>`__ - example of policies
- `image_reconstruction.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/image_reconstruction.ipynb>`__ - example of image reconstruction
- `classification_and_regression.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/classification_and_regression.ipynb>`__ - example of multitask learning with classification and regression
- `semantic_segmentation.ipynb <https://github.com/GRAAL-Research/poutyne/blob/master/examples/semantic_segmentation.ipynb>`__ - example of semantic segmentation

or in ``Google Colab``:

- `introduction_pytorch_poutyne.ipynb <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/introduction_pytorch_poutyne.ipynb>`__ (`tutorial version <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/tutorials/introduction_pytorch_poutyne_tutorial.ipynb>`__) - comparison of Poutyne with bare PyTorch and usage examples of Poutyne callbacks and the ModelBundle class.
- `tips_and_tricks.ipynb <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/tips_and_tricks.ipynb>`__ -  tips and tricks using Poutyne.
- `transfer_learning.ipynb <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/transfer_learning.ipynb>`__ - transfer learning on ``ResNet-18`` on the `CUB-200 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`__ dataset.
- `policy_cifar_example.ipynb <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/policy_cifar_example.ipynb>`__ - policies API, FastAI-like learning rate policies
- `policy_interface.ipynb <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/policy_interface.ipynb>`__ - example of policies
- `image_reconstruction.ipynb <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/image_reconstruction.ipynb>`__ - example of image reconstruction
- `classification_and_regression.ipynb <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/classification_and_regression.ipynb>`__ - example of multitask learning with classification and regression
- `semantic_segmentation.ipynb <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/semantic_segmentation.ipynb>`__ - example of semantic segmentation

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

Poutyne's name comes from `poutine <https://en.wikipedia.org/wiki/Poutine>`__, the well-known dish from Quebec. It is usually composed of French fries, squeaky cheese curds and brown gravy. However, in Quebec, it also has the meaning of something that is an `"ordinary or common subject or activity" <https://fr.wiktionary.org/wiki/poutine>`_. Thus, Poutyne will get rid of the ordinary boilerplate code that plain `PyTorch <https://pytorch.org/>`_ training usually entails.

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
  examples/sequence_tagging
  examples/policy_interface
  examples/transfer_learning
  examples/image_reconstruction
  examples/classification_and_regression
  examples/semantic_segmentation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
