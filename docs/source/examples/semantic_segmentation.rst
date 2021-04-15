.. role:: hidden
    :class: hidden-section

Semantic segmentation using Poutyne
***********************************

.. note::

    - See the notebook `here <https://github.com/GRAAL-Research/poutyne/blob/master/examples/semantic_segmentation.ipynb>`_
    - Run in `Google Colab <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/semantic_segmentation.ipynb>`_

Semantic segmentation refers to the process of linking each pixel in an image to a class label. We can think of semantic segmentation as image classification at a pixel level. The image below clarifies the definition of semantic segmentation.

.. image:: /_static/img/semantic_segmentation/semantic_segmentation.png

`Source <https://www.jeremyjordan.me/semantic-segmentation/>`_

In this example, we will use and train a convolutional U-Net to design a network for semantic segmentation. In other words, we formulate the task of semantic segmentation as an image translation problem. We download and use the VOCSegmentation 2007 dataset for this purpose.

U-Net is a convolutional neural network similar to convolutional autoencoders. However, U-Net takes advantage of shortcuts between the encoder (contraction path) and decoder (expanding path), which helps it handle the vanishing gradient problem. In the following sections, we will install and import the segmentation-models-Pytorch library, which contains different U-Net architectures.

.. code-block:: python

    %pip install segmentation-models-pytorch 
    
Let’s import all the needed packages.

.. code-block:: python

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import torchvision.transforms as tfms
    import torchvision.datasets as datasets
    import segmentation_models_pytorch as smp
    from poutyne import Model, ModelCheckpoint, CSVLogger, set_seeds
    from torch.utils.data import DataLoader
    from torchvision.utils import  make_grid
    
Training constants
==================

.. code-block:: python

    learning_rate = 0.0005
    batch_size = 32
    image_size = 224
    num_epochs = 70
    imagenet_mean = [0.485, 0.456, 0.406]  # mean of the imagenet dataset for normalizing 
    imagenet_std = [0.229, 0.224, 0.225]  # std of the imagenet dataset for normalizing 
    set_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('The current processor is ...', device)

Loading the VOCSegmentation dataset    
===================================

The VOCSegmentation dataset can be easily downloaded from ``torchvision.datasets``. This dataset allows you to apply the needed transformations on the ground-truth directly and define the proper transformations for the input images. To do so, we use the ``target_transfrom`` argument and set it to your transformation function of interest.

.. code-block:: python

    input_transform = tfms.Compose([
        tfms.Resize((image_size, image_size)),  
        tfms.ToTensor(),        
        tfms.Normalize(imagenet_mean, imagenet_std)
    ])
    
    target_transform = tfms.Compose([
        tfms.Resize((image_size, image_size)),  
        tfms.ToTensor(),        
    ])
    
    # Creating the dataset
    train_dataset = datasets.VOCSegmentation('./datasets/', year='2007', download=True, image_set='train', transform=input_transform, target_transform=target_transform)
    valid_dataset = datasets.VOCSegmentation('./datasets/', year='2007', download=True, image_set='val', transform=input_transform, target_transform=target_transform)
    
    # Creating the dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

A random batch of the VODSegmentation dataset images
====================================================

Let's see some of the input samples inside the training dataset.

.. code-block:: python

    samples = next(iter(valid_dataloader))
    inputs = samples[0]
    input_grid = make_grid(inputs)
    
    fig = plt.figure(figsize=(10, 10))
    input_grid = input_grid.numpy()
    input_grid = input_grid.transpose((1, 2, 0)) * imagenet_std + imagenet_mean
    inp = np.clip(input_grid, 0, 1)
    plt.imshow(inp)
    plt.axis('off')
    plt.show()
    
.. image:: /_static/img/semantic_segmentation/voc_segment_batch.png 

The ground-truth (segmentation map) for the image grid shown above is as below.

.. code-block:: python

    ground_truth = samples[1]
    input_grid = make_grid(ground_truth)
    inp = input_grid.numpy()[0]
    
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(inp)
    plt.axis('off')
    plt.show()
    
.. image:: /_static/img/semantic_segmentation/voc_segment_batch_gt.png 

It is worth mentioning that, as we have approached the segmentation task as an image translation problem, we take advantage of MSELoss for the training. Moreover, we believe that using the U-Net with a pre-trained encoder would help the network converge sooner and better. As this convolutional encoder is previously trained on the ImageNet, it is able to recognize low-level features (such as edge, color, etc.) and high-level features at its first and final layers, respectively.

.. code-block:: python

    # specifying loss function
    criterion = nn.MSELoss()
    
    # specifying the network
    network = smp.Unet('resnet34', encoder_weights='imagenet')
    
    # specifying optimizer
    optimizer = optim.Adam (network.parameters(), lr=learning_rate)  

As noticed in the section above, the ResNet-34-U-Net network is imported from the segmentation-models-pytorch library which contains many other architectures as well. You can import and use other available networks to try to increase the accuracy.

Training deep neural networks is a challenging task, especially when we are dealing with data with big sizes or numbers. There are numerous factors and hyperparameters which play an important role in the success of the network. One of these determining factors is the number of epochs. The right number of epochs would help your network train well. However, lower and higher numbers would make your network underfit or overfit, respectively. With some data types (such as images or videos), it is very time-consuming to repeat the training for different numbers of epochs to find the best one. Poutyne library has provided some fascinating tools to address this problem.

As you would notice in the following sections, by the use of `callbacks <https://poutyne.org/callbacks.html>`_, you would be able to record and retrieve the best parameters (weights) through your rather big number of epochs without needing to repeat the training process again and again. Moreover, Poutyne also gives you the possibility to resume your training from the last done epoch if you feel the need for even more iterations.

.. code-block:: python

    #callbacks
    
    save_path = 'saves'
    
    # Creating saving directory 
    os.makedirs(save_path, exist_ok=True)
    
    callbacks = [
        # Save the latest weights to be able to continue the optimization at the end for more epochs.
        ModelCheckpoint(os.path.join(save_path, 'last_weights.ckpt')),
    
        # Save the weights in a new file when the current model is better than all previous models.
        ModelCheckpoint(os.path.join(save_path, 'best_weight.ckpt'),
                        save_best_only=True, restore_best=True, verbose=True),
    
        # Save the losses for each epoch in a TSV.
        CSVLogger(os.path.join(save_path, 'log.tsv'), separator='\t'),
    ]

Training
========

.. code-block:: python

    # Poutyne Model on GPU
    model = Model(network, optimizer, criterion, device=device)
    
    # Train
    model.fit_generator(train_dataloader, valid_dataloader, epochs=num_epochs, callbacks=callbacks)

Calculation of the scores and visualization of results
======================================================

There is one more helpful feature in Poutyne, which makes the evaluation task more easy and straight forward. Usually, computer vision researchers try to evaluate their trained networks on validation/test datasets by obtaining the scores (accuracy or loss usually), ground truths, and computed results simultaneously. The ``evaluate`` methods in Poutyne provides you with the scores but also have made the other two items ready for further analysis and visualization. In the next few blocks of code, you will see some examples.

.. code-block:: python

    loss, predictions, ground_truth = model.evaluate_generator(valid_dataloader, return_pred=True, return_ground_truth=True)

We show some of the segmentation results in the image below (grayscale):

.. code-block:: python

    outputs = torch.tensor(model.predict_on_batch(inputs))
    output_grid = make_grid(outputs)
    out = output_grid.numpy().transpose((1, 2, 0))
    out = np.clip(out, 0, 1)
    
    fig = plt.figure(figsize=(10, 10))
    plt.imshow((out))
    plt.show()

.. image:: /_static/img/semantic_segmentation/segment_out.png 

Here, we show one of the input samples along with its segmentation ground truth and the produced output.

.. code-block:: python

    sample_number = 14
    
    input_sample = inputs[sample_number].numpy().transpose((1, 2, 0)) * imagenet_std + imagenet_mean
    ground_truth_sample = ground_truth[sample_number][0]
    output_sample = outputs[sample_number][0].numpy()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(input_sample)
    ax1.axis('off')
    ax1.set_title('input')
    
    ax2.imshow(ground_truth_sample)
    ax2.axis('off')
    ax2.set_title('GT')
    
    ax3.imshow(output_sample)
    ax3.axis('off')
    ax3.set_title('output')
    plt.show()

.. image:: /_static/img/semantic_segmentation/segment_compare.png 

Last note
=========

This example shows you how to design and train your own segmentation network simply. However, to get better results, you can play with the hyperparameters and do further finetuning to increase the accuracy.
