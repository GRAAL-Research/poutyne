![Poutyne Logo](https://raw.githubusercontent.com/GRAAL-Research/poutyne/master/docs/source/_static/logos/poutyne-dark.png)

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](http://www.gnu.org/licenses/lgpl-3.0)
[![Continuous Integration](https://github.com/GRAAL-Research/poutyne/workflows/Continuous%20Integration/badge.svg)](https://github.com/GRAAL-Research/poutyne/actions?query=workflow%3A%22Continuous+Integration%22+branch%3Amaster)
[![codecov](https://codecov.io/gh/GRAAL-Research/poutyne/branch/master/graph/badge.svg?token=H8D1nZ1wTR)](https://codecov.io/gh/GRAAL-Research/poutyne)

## Here is Poutyne.

Poutyne is a simplified framework for [PyTorch](https://pytorch.org/) and handles much of the boilerplating code needed to train neural networks.

Use Poutyne to:
- Train models easily.
- Use callbacks to save your best model, perform early stopping and much more.

Read the documentation at [Poutyne.org](https://poutyne.org).

Poutyne is compatible with  the __latest version of PyTorch__ and  __Python >= 3.6__.

### Cite
```
@misc{Paradis_Poutyne_A_Simplified_2020,
    author = {Paradis, Frédérik and Beauchemin, David and Godbout, Mathieu and Alain, Mathieu and Garneau, Nicolas and Otte, Stefan and Tremblay, Alexis and Bélanger, Marc-Antoine and Laviolette, François},
    title  = {{Poutyne: A Simplified Framework for Deep Learning}},
    year   = {2020},
    url    = {https://poutyne.org}
}
```


------------------


## Getting started: few seconds to Poutyne

The core data structure of Poutyne is a [Model](poutyne/framework/model.py), a way to train your own [PyTorch](https://pytorch.org/docs/master/nn.html) neural networks.

How Poutyne works is that you create your [PyTorch](https://pytorch.org/docs/master/nn.html) module (neural network) as usual but when comes the time to train it you feed it into the Poutyne Model, which handles all the steps, stats and callbacks, similar to what [Keras](https://keras.io) does.

Here is a simple example:

```python
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
```

Select a PyTorch device so that it runs on GPU if you have one:

```python
cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")
```

Create yourself a [PyTorch](https://pytorch.org/docs/master/nn.html) network:

```python
network = nn.Sequential(
    nn.Linear(num_features, hidden_state_size),
    nn.ReLU(),
    nn.Linear(hidden_state_size, num_classes)
)
```

You can now use Poutyne's model to train your network easily:

```python
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
```

Since Poutyne is inspired by [Keras](https://keras.io), one might have notice that this is really similar to some of its [functions](https://keras.io/models/model/).

You can evaluate the performances of your network using the ``evaluate`` method of Poutyne's model:

```python
loss, (accuracy, f1score) = model.evaluate(test_x, test_y)
```

Or only predict on new data:

```python
predictions = model.predict(test_x)
```

[See the complete code here.](https://github.com/GRAAL-Research/poutyne/blob/master/examples/basic_random_classification.py) Also, [see this](https://github.com/GRAAL-Research/poutyne/blob/master/examples/basic_random_regression.py) for an example for regression.

One of the strengths Poutyne are [callbacks](https://poutyne.org/callbacks.html). They allow you to save checkpoints, log training statistics and more. See this [notebook](https://github.com/GRAAL-Research/poutyne/blob/master/examples/introduction_pytorch_poutyne.ipynb) for an introduction to callbacks. In that vein, Poutyne also offers an [ModelBundle class](https://poutyne.org/experiment.html#poutyne.ModelBundle) that offers automatic checkpointing, logging and more using callbacks under the hood. Here is an example of usage.

```python
from poutyne import ModelBundle

# Everything is saved in ./saves/my_classification_network
model_bundle = ModelBundle.from_network(
    './saves/my_classification_network', network, optimizer='sgd', task='classif', device=device
)

model_bundle.train_data(train_x, train_y, validation_data=(valid_x, valid_y), epochs=5)

model_bundle.test_data(test_x, test_y)
```

[See the complete code here.](https://github.com/GRAAL-Research/poutyne/blob/master/examples/basic_random_classification_with_model_bundle.py) Also, [see this](https://github.com/GRAAL-Research/poutyne/blob/master/examples/basic_random_regression_with_model_bundle.py) for an example for regression.


------------------

## Installation

Before installing Poutyne, you must have the latest version of [PyTorch](https://pytorch.org/) in your environment.

- **Install the stable version of Poutyne:**

```sh
pip install poutyne
```

- **Install the latest development version of Poutyne:**

```sh
pip install -U git+https://github.com/GRAAL-Research/poutyne.git@dev
```

- **Install and develop on top of the provided Docker Image**

```sh
docker pull ghcr.io/graal-research/poutyne/poutyne:latest
```

------------------

## Learning Material

### Blog posts

* [Medium PyTorch post](https://medium.com/pytorch/poutyne-a-simplified-framework-for-deep-learning-in-pytorch-74b1fc1d5a8b) - Presentation of the basics of Poutyne and how it can help you be more efficient when developing neural networks with PyTorch.

### Examples

Look at notebook files with full working [examples](https://github.com/GRAAL-Research/poutyne/blob/master/examples/):

* [introduction.ipynb](https://github.com/GRAAL-Research/poutyne/blob/master/examples/introduction.ipynb) ([tutorial version](https://github.com/GRAAL-Research/poutyne/blob/master/tutorials/introduction_pytorch_poutyne_tutorial.ipynb)) - comparison of Poutyne with bare PyTorch and usage examples of Poutyne callbacks and the ModelBundle class.
* [tips_and_tricks.ipynb](https://github.com/GRAAL-Research/poutyne/blob/master/examples/tips_and_tricks.ipynb) - tips and tricks using Poutyne
* [sequence_tagging.ipynb](https://github.com/GRAAL-Research/poutyne/blob/master/examples/sequence_tagging.ipynb) - Sequence tagging with an RNN
* [transfer_learning.ipynb](https://github.com/GRAAL-Research/poutyne/blob/master/examples/transfer_learning.ipynb) - transfer learning on `ResNet-18` on the [CUB-200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset.
* [policy_interface.ipynb](https://github.com/GRAAL-Research/poutyne/blob/master/examples/policy_interface.ipynb) - example of policies
* [image_reconstruction.ipynb](https://github.com/GRAAL-Research/poutyne/blob/master/examples/image_reconstruction.ipynb) - example of image reconstruction
* [classification_and_regression.ipynb](https://github.com/GRAAL-Research/poutyne/blob/master/examples/classification_and_regression.ipynb) - example of multitask learning with classification and regression
* [semantic_segmentation.ipynb](https://github.com/GRAAL-Research/poutyne/blob/master/examples/semantic_segmentation.ipynb) - example of semantic segmentation

or in ``Google Colab``:

* [introduction.ipynb](https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/introduction.ipynb) ([tutorial version](https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/tutorials/introduction_pytorch_poutyne_tutorial.ipynb)) - comparison of Poutyne with bare PyTorch and usage examples of Poutyne callbacks and the ModelBundle class.
* [tips_and_tricks.ipynb](https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/tips_and_tricks.ipynb) - tips and tricks using Poutyne
* [sequence_tagging.ipynb](https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/sequence_tagging.ipynb) - Sequence tagging with an RNN
* [transfer_learning.ipynb](https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/transfer_learning.ipynb) - transfer learning on `ResNet-18` on the [CUB-200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) dataset.
* [policy_interface.ipynb](https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/policy_interface.ipynb) - example of policies
* [image_reconstruction.ipynb](https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/image_reconstruction.ipynb) - example of image reconstruction
* [classification_and_regression.ipynb](https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/classification_and_regression.ipynb) - example of multitask learning with classification and regression
* [semantic_segmentation.ipynb](https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/semantic_segmentation.ipynb) - example of semantic segmentation

### Videos

* [Presentation on Poutyne](https://youtu.be/gQ3SW5r7HSs) given at one of the weekly presentations of the Institute Intelligence and Data (IID) of Université Laval. [Slides](https://github.com/GRAAL-Research/poutyne/blob/master/slides/poutyne.pdf) and the [associated Latex source code](https://github.com/GRAAL-Research/poutyne/blob/master/slides/src/) are also available.

------------------

## Contributing to Poutyne

We welcome user input, whether it is regarding bugs found in the library or feature propositions ! Make sure to have a look at our [contributing guidelines](https://github.com/GRAAL-Research/poutyne/blob/master/CONTRIBUTING.md) for more details on this matter.

------------------

## License

Poutyne is LGPLv3 licensed, as found in the [LICENSE file](https://github.com/GRAAL-Research/poutyne/blob/master/LICENSE).

------------------

## Why this name, Poutyne?

Poutyne's name comes from [poutine](https://en.wikipedia.org/wiki/Poutine), the well-known dish from Quebec. It is usually composed of French fries, squeaky cheese curds and brown gravy. However, in Quebec, poutine also has the meaning of something that is an ["ordinary or common subject or activity"](https://fr.wiktionary.org/wiki/poutine). Thus, Poutyne will get rid of the ordinary boilerplate code that plain [PyTorch](https://pytorch.org) training usually entails.

![Poutine](https://upload.wikimedia.org/wikipedia/commons/4/4e/La_Banquise_Poutine_%28cropped%29.jpg)
*Yuri Long from Arlington, VA, USA \[[CC BY 2.0](https://creativecommons.org/licenses/by/2.0)\]*

------------------
