.. role:: hidden
    :class: hidden-section

Image reconstruction using Poutyne
***********************************

.. note::

    - See the notebook `here <https://github.com/GRAAL-Research/poutyne/blob/master/examples/image_reconstruction.ipynb>`_
    - Run in `Google Colab <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/image_reconstruction.ipynb>`_

In this example, we train a simple convolutional autoencoder (Conv-AE) on the MNIST dataset to learn image reconstruction. The Conv-AE is composed of two parts: an encoder and a decoder. The encoder encodes the input images to extract compact image features. The decoder, on the other hand, decodes the extracted features to reconstruct the input images.

.. image:: /_static/img/image_reconstruction/AE.png

Let's import all the needed packages.

.. code-block:: python

    import numpy as np
    import math
    import os
    import matplotlib.pyplot as plt
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset, Dataset
    
    import torchvision
    import torchvision.datasets as datasets
    import torchvision.transforms as tfms
    from torchvision.utils import make_grid

    import poutyne
    from poutyne import set_seeds, Model, Experiment, ModelCheckpoint, CSVLogger

Training constants
==================

.. code-block:: python

    num_epochs = 3
    learning_rate = 0.001
    batch_size = 32
    image_size = 224
    valid_split_percent = 0.2
    momentum = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('The running processor is...', device)
    set_seeds(42)

Loading the MNIST dataset
=========================

The MNIST dataset is directly downloaded from the ``torchvision.datasets`` package. The training dataset contains 60,000 images of digits of size 28x28. However, we separate 20% of the full train dataset as a validation dataset. On the other hand, by setting the "train" argument to ``False``, the test dataset containing 10,000 images is downloaded and saved in the "datasets" directory.

.. code-block:: python

    full_train_dataset = datasets.MNIST('./datasets/', train=True, download=True, transform=tfms.ToTensor())
    test_dataset = datasets.MNIST('./datasets/', train=False, download=True, transform=tfms.ToTensor())
    
    # Selecting and seperating a proportion of the full_train_dataset to create the validation dataset.
    full_dataset_length = len(full_train_dataset)
    indices = list(np.arange(full_dataset_length))
    np.random.shuffle(indices)
    train_indices = indices[math.floor(full_dataset_length * valid_split_percent):]
    valid_indices = indices[:math.floor(full_dataset_length * valid_split_percent)]
    
    train_dataset = Subset(full_train_dataset, train_indices)
    valid_dataset = Subset(full_train_dataset, valid_indices)

The format of the downloaded MNIST dataset is proper for a classification application, which means each sample is composed of an image and a label (the digit drawn in the image). However, for image reconstruction, the dataset should contain an input image and a target image, which simply are the same. Hence, using the code below, we define a new dataset that wraps an MNIST dataset and provides an image as an input and sets that image as its target, as well. In other words, we change the format of each dataset sample, from (image, label) to the (image, image).

.. code-block:: python

    class ImageReconstructionDataset(Dataset):
    
        def __init__(self, dataset):
            self.dataset = dataset
              
        def __getitem__(self, index):
            input_image = self.dataset[index][0]
            reconstructed_image = input_image  # In image reconstruction, input and target images are the same.
              
            return input_image, reconstructed_image
          
        def __len__(self):
              
            return len(self.dataset)  

Finally, in the section below, we wrap the MNIST datasets into our wrapper and create dataloaders for them.

.. code-block:: python

    train_dataset_new = ImageReconstructionDataset(train_dataset)
    valid_dataset_new = ImageReconstructionDataset(valid_dataset)
    test_dataset_new = ImageReconstructionDataset(test_dataset)
    
    train_dataloader = DataLoader(train_dataset_new, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset_new, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset_new, batch_size=1, shuffle=False)

Convolutional autoencoder
=========================

.. code-block:: python

    class ConvAutoencoder(nn.Module):
        def __init__(self):
            super(ConvAutoencoder, self).__init__()
          
            #encoder
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
            self.conv2 = nn.Conv2d(32, 4, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
          
            #decoder
            self.t_conv1 = nn.ConvTranspose2d(4, 32, 2, stride=2)
            self.t_conv2 = nn.ConvTranspose2d(32, 1, 2, stride=2)
    
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)  # compressed representation
            x = torch.relu(self.t_conv1(x))
            x = torch.sigmoid(self.t_conv2(x))
                  
            return x
    
    network = ConvAutoencoder()
    
In order to interact with the optimization process, `callbacks <https://poutyne.org/callbacks.html>`_ are defined and added to a list using the code below. They will save the last weights, best weights and the logs, respectively.

.. code-block:: python

    save_path = 'saves'
    
    # Creating saving directory 
    os.makedirs(save_path, exist_ok=True)
    
    callbacks = [
        # Save the latest weights to be able to continue the optimization at the end for more epochs.
        ModelCheckpoint(os.path.join(save_path, 'last_weights.ckpt')),
    
        # Save the weights in a new file when the current model is better than all previous models.
        ModelCheckpoint(os.path.join(save_path, 'best_weight.ckpt'),
                        save_best_only=True, verbose=True),
    
        # Save the losses for each epoch in a TSV.
        CSVLogger(os.path.join(save_path, 'log.tsv'), separator='\t'),
    ]

Let's specify the loss and the optimization function.

.. code-block:: python

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

Training
========

.. code-block:: python

    # Poutyne Model on GPU
    model = Model(network, optimizer, criterion, device=device)
    
    # Train
    model.fit_generator(train_dataloader, valid_dataloader, epochs=num_epochs, callbacks=callbacks)

A random batch of the MNIST dataset images
==========================================

Let's see some of the input samples inside the training dataset.

.. code-block:: python

    inputs = next(iter(train_dataloader))[0]
    input_grid = make_grid(inputs)
    fig = plt.figure(figsize=(10, 10))
    inp = input_grid.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.show()

.. image:: /_static/img/image_reconstruction/mnist_batch.png

Reconstructed images after 3 epochs of training
===============================================

In order to visually evaluate the quality of the results, here, we show the reconstruction results of the samples shown above.

.. code-block:: python

    # Calculating predictions of the trained network on a batch
    outputs = torch.tensor(model.predict_on_batch(inputs))
    output_grid = make_grid(outputs)
    fig = plt.figure(figsize=(10, 10))
    out = output_grid.numpy().transpose((1, 2, 0))
    plt.imshow(out)
    plt.show()

.. image:: /_static/img/image_reconstruction/mnist_3epoch.png

Evaluation
==========

One of the strong and useful tools in Poutyne is the ``evaluate`` methods, which not only provide you with the evaluation metrics but also provide the ground truths and the predictions if the related arguments have been set to ``True`` (as below).

.. code-block:: python

    # evaluating the trained network on test data
    loss, predictions, ground_truth = model.evaluate_generator(test_dataloader, return_pred=True, return_ground_truth=True)

In most computer vision applications, such as image reconstruction, it is very important to check the network's failures (or abilities, vice versa). The following part shows an input and a reconstructed image, as well as its reconstruction error map. The reconstruction error map shows which part of the image has not been reconstructed accurately.

.. code-block:: python

    sample_number = 2   # a sample from test dataset
    
    sample = ground_truth[sample_number][0]
    sample_prediction_result_3epochs = predictions[sample_number][0]
    
    recunstruction_error_map_3epochs = sample - sample_prediction_result_3epochs  #reconstruction error map
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(sample)
    ax1.set_title('sample')
    
    ax2.imshow(sample_prediction_result_3epochs)
    ax2.set_title('prediction')
    
    ax3.imshow(np.abs(recunstruction_error_map_3epochs))
    ax3.set_title('reconstruction error')
    plt.show()

.. image:: /_static/img/image_reconstruction/rec_error_3epoch.png

Resuming the training for more epochs
=====================================

If we find the past epochs not enough, Poutyne allows you to resume the training from the last done epoch, as below. Please note that, in the ``callbacks`` that we defined before, since we did not set the ``restore_best`` argument in ``ModelCheckpoint`` to ``True``, our model stays at the last epoch after finishing the first part of the training. Hence, by setting the ``initial_epoch`` to the last epoch of the previous training, we can resume our training for more epochs, using the last state of the neural network.
    
.. code-block:: python    
    
    model.fit_generator(train_dataloader, valid_dataloader, epochs=13, callbacks=callbacks, initial_epoch=num_epochs)    

Reconstructed images after the second training process
======================================================

Now let's visualize the quality of the results after the second phase of training.

.. code-block:: python  

    outputs = torch.tensor(model.predict_on_batch(inputs))
    output_grid = make_grid(outputs)
    fig = plt.figure(figsize=(10, 10))
    out = output_grid.numpy().transpose((1, 2, 0))
    plt.imshow(out)
    plt.show()

.. image:: /_static/img/image_reconstruction/mnist_13epoch.png

.. code-block:: python

    loss, predictions, ground_truth = model.evaluate_generator(test_dataloader, return_pred=True, return_ground_truth=True)

Here, we compare the reconstruction accuracy of the network after 3 epochs and 13 epochs of training.

.. code-block:: python

    sample_number = 2 
    sample = ground_truth[sample_number][0]
    sample_prediction_result_13epochs = predictions[sample_number][0]
    recunstruction_error_map_13epochs = sample - sample_prediction_result_13epochs  #reconstruction error map
    
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    axs[0, 0].imshow(sample)
    axs[0, 0].set_title('sample')
    
    axs[0, 1].imshow(sample_prediction_result_3epochs)
    axs[0, 1].set_title('prediction')
    
    axs[0, 2].imshow(np.abs(recunstruction_error_map_3epochs))
    axs[0, 2].set_title('rec_error epoch3')
    
    axs[1, 0].imshow(sample)
    axs[1, 0].set_title('sample')
    
    axs[1, 1].imshow(sample_prediction_result_13epochs)
    axs[1, 1].set_title('prediction')
    
    axs[1, 2].imshow(np.abs(recunstruction_error_map_13epochs))
    axs[1, 2].set_title('rec_error epoch13')
    
    plt.show()

.. image:: /_static/img/image_reconstruction/mnist_compare.png

You can also try more finetuning, by changing the hyperparameters (network capacity, epochs, etc) to increase the accuracy as much as you want.
