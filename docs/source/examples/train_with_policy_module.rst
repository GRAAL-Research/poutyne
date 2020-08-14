.. role:: hidden
    :class: hidden-section

Train CIFAR with the ``policy`` module
**************************************

.. note:: See the notebook `here <https://github.com/GRAAL-Research/poutyne/blob/master/examples/policy_cifar_example.ipynb>`_

Let's import all the needed packages.

.. code-block:: python

    import torch
    import torchvision.datasets as datasets
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18
    import torch.nn as nn
    import torch.optim as optim
    from poutyne.framework import Model
    from poutyne.framework import OptimizerPolicy, one_cycle_phases



But first, let's set the training constants, the CUDA device used for training if one is present, we set the batch size (i.e. the number of elements to see before updating the model) and the number of epochs (i.e. the number of times we see the full dataset).

.. code-block:: python

    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

    batch_size = 1024
    epochs = 5

Load the data
=============

.. code-block:: python

    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]


    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.3, .3, .3),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std),
    ])

.. code-block:: python

    root = "data"
    train_ds = datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
    val_ds = datasets.CIFAR10(root, train=False, transform=val_transform, download=True)

.. code-block:: python

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )


The model
=========

We'll train a simple ``resNet-18`` network.
This takes a while without GPU but is pretty quick with GPU.

.. code-block:: python

    def get_module():
        model = resnet18(pretrained=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 10)
        return model

Training without the ``policies`` module
========================================

.. code-block:: python

    pytorch_network = get_module().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(pytorch_network.parameters(), lr=0.01)

    model = Model(
        pytorch_network,
        optimizer,
        criterion,
        batch_metrics=["acc"],
    )
    model = model.to(device)

    history = model.fit_generator(
        train_dl,
        val_dl,
        epochs=epochs,
    )


Training with the ``policies`` module
=====================================

.. code-block:: python

    steps_per_epoch = len(train_dl)
    steps_per_epoch

.. code-block:: python

    pytorch_network = get_module().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(pytorch_network.parameters(), lr=0.01)

    model = Model(
        pytorch_network,
        optimizer,
        criterion,
        batch_metrics=["acc"],
    )
    model = model.to(device)

    policy = OptimizerPolicy(
        one_cycle_phases(epochs * steps_per_epoch, lr=(0.01, 0.1, 0.008)),
    )
    history = model.fit_generator(
        train_dl,
        val_dl,
        epochs=epochs,
        callbacks=[policy],
    )

