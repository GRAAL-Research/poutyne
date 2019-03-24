# Poutyne: Deep Learning framework for [PyTorch](http://pytorch.org/)

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)
[![Build Status](https://travis-ci.org/GRAAL-Research/poutyne.svg?branch=master)](https://travis-ci.org/GRAAL-Research/poutyne)

## Here is Poutyne.

> As you can see, PyToune has changed its name for Poutyne. From now on, please use the new package name `poutyne`. The `pytoune` package has been kept for this release but will be removed in the next release.

Poutyne is a Keras-like framework for [PyTorch](https://pytorch.org/) and handles much of the boilerplating code needed to train neural networks.

Use Poutyne to:
- Train models easily.
- Use callbacks to save your best model, perform early stopping and much more.

Read the documentation at [Poutyne.org](https://poutyne.org).

Poutyne is compatible with  the __latest version of PyTorch__ and  __Python >= 3.5__.

### Cite
```
@misc{frederikParadisPoutyne,
  author = {Paradis, Fr{\'e}d{\'e}rik and Garneau, Nicolas},
  title  = {{Poutyne}: Keras-like framework for {PyTorch}},
  year   = {2018--},
  url    = {\url{https://poutyne.org}}
}
```


------------------


## Getting started: few seconds to Poutyne

The core data structure of Poutyne is a ``Model``, a way to train your own [PyTorch](https://pytorch.org/docs/master/nn.html) neural networks.

How Poutyne works is that you create your [PyTorch](https://pytorch.org/docs/master/nn.html) module (neural network) as usual but when comes the time to train it you feed it into the Poutyne Model, which handles all the steps, stats and callbacks, similar to what [Keras](https://keras.io) does.

Here is a simple example:

```python
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
```

Create yourself a [PyTorch](https://pytorch.org/docs/master/nn.html) network;

```python
pytorch_module = torch.nn.Linear(num_features, num_classes)
```

You can now use Poutyne's model to train your network easily;

```python
model = Model(pytorch_module, 'sgd', 'cross_entropy', metrics=['accuracy'])
model.fit(
    train_x, train_y,
    validation_x=valid_x,
    validation_y=valid_y,
    epochs=5,
    batch_size=32
  )
```

This is really similar to the ``model.compile`` function as in [Keras](https://keras.io);

```python
# Keras way to compile and train
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=5, batch_size=32)
```

You can evaluate the performances of your network using the ``evaluate`` method of Poutyne's model;

```python
loss_and_metrics = model.evaluate(test_x, test_y)
```

Or only predict on new data;

```python
predictions = model.predict(test_x)
```

As you can see, Poutyne is inspired a lot by the friendliness of [Keras](https://keras.io). See the Poutyne documentation at [Poutyne.org](https://poutyne.org) for more.


------------------


## Installation

Before installing Poutyne, you must have the latest version of [PyTorch](https://pytorch.org/) in your environment.

- **Install the stable version of Poutyne:**

```sh
pip install poutyne
```

- **Install the latest version of Poutyne:**

```sh
pip install -U git+https://github.com/GRAAL-Research/poutyne.git
```

------------------

## Why this name, Poutyne?

Poutyne (or poutine in Québécois) is now the well-known dish from Quebec composed of French fries, squeaky cheese curds and brown gravy. However, in Quebec, it also has the meaning of something that is an ["ordinary or common subject or activity"](https://fr.wiktionary.org/wiki/poutine). Thus, Poutyne will get rid of the ordinary boilerplate code that plain [PyTorch](https://pytorch.org) training usually entails.

![Poutine](https://upload.wikimedia.org/wikipedia/commons/4/4e/La_Banquise_Poutine_%28cropped%29.jpg)
*Yuri Long from Arlington, VA, USA \[[CC BY 2.0](https://creativecommons.org/licenses/by/2.0)\]*

------------------
