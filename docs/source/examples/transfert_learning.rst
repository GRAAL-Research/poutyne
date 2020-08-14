.. role:: hidden
    :class: hidden-section

Transfer learning example
**************************
.. note:: See the notebook `here <https://github.com/GRAAL-Research/poutyne/blob/master/examples/transfert_learning.ipynb>`_

But first, let's import all the needed packages.

.. code-block:: python

    import math
    import os
    import tarfile
    import urllib.request
    from shutil import copyfile

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    from torch.utils import model_zoo
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    from poutyne import set_seeds
    from poutyne.framework import Model, ModelCheckpoint, CSVLogger


Also, we need to set Pythons's, NumPy's and PyTorch's seeds by using Poutyne function so that our training is (almost) reproducible.

.. code-block:: python

    set_seeds(42)


.. code-block:: python

    def download_and_extract_dataset(path):
        tgz_filename = "images.tgz"
        urllib.request.urlretrieve("http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz", tgz_filename)
        os.makedirs(path, exist_ok=True)
        archive = tarfile.open(tgz_filename)
        archive.extractall(path)

.. code-block:: python

    def copy(source_path, filenames, dest_path):
        for filename in filenames:
            source = os.path.join(source_path, filename)
            dest = os.path.join(dest_path, filename)
            copyfile(source, dest)

    def split_train_valid_test(dataset_path, train_path, valid_path, test_path, train_split=0.6, valid_split=0.2): # test_split=0.2
        for classname in sorted(os.listdir(dataset_path)):
            if classname.startswith('.'):
                continue
            train_class_path = os.path.join(train_path, classname)
            valid_class_path = os.path.join(valid_path, classname)
            test_class_path = os.path.join(test_path, classname)

            os.makedirs(train_class_path, exist_ok=True)
            os.makedirs(valid_class_path, exist_ok=True)
            os.makedirs(test_class_path, exist_ok=True)

            dataset_class_path = os.path.join(dataset_path, classname)
            filenames = sorted(filename for filename in os.listdir(dataset_class_path) if not filename.startswith('.'))
            np.random.shuffle(filenames)

            num_examples = len(filenames)
            train_last_idx = math.ceil(num_examples*train_split)
            valid_last_idx = train_last_idx + math.floor(num_examples*valid_split)
            train_filenames = filenames[0:train_last_idx]
            valid_filenames = filenames[train_last_idx:valid_last_idx]
            test_filenames = filenames[valid_last_idx:]
            copy(dataset_class_path, train_filenames, train_class_path)
            copy(dataset_class_path, valid_filenames, valid_class_path)
            copy(dataset_class_path, test_filenames, test_class_path)


We do the split train/valid/test.

.. code-block:: python

    base_path = './CUB200'
    dataset_path = os.path.join(base_path, 'images')
    train_path = os.path.join(base_path, 'train')
    valid_path = os.path.join(base_path, 'valid')
    test_path = os.path.join(base_path, 'test')

.. code-block:: python

    download_and_extract_dataset(base_path)
    split_train_valid_test(dataset_path, train_path, valid_path, test_path)


Now, let's set our training constants. We first have the CUDA device used for training if one is present. Second, we set the number of classes (i.e. one for each number). Finally, we set the batch size (i.e. the number of elements to see before updating the model), the learning rate for the optimizer, and the number of epochs (i.e. the number of times we see the full dataset).

.. code-block:: python

    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

    num_classes = 200
    batch_size = 32
    learning_rate = 0.1
    n_epoch = 30


Creation of the PyTorch's datasets for our problem.

.. code-block:: python

    norm_coefs = {}
    norm_coefs['cub200'] = [(0.47421962,  0.4914721 ,  0.42382449), (0.22846779,  0.22387765,  0.26495799)]
    norm_coefs['imagenet'] = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(*norm_coefs['cub200'])
    ])

    train_set = ImageFolder(train_path, transform=transform)
    valid_set = ImageFolder(valid_path, transform=transform)
    test_set = ImageFolder(test_path, transform=transform)


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)


We load a pretrained ``ResNet-18`` networks and replace the head with the number of neurons equal to our number of classes.

.. code-block:: python

    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)


We freeze the network except for its head.

.. code-block:: python

    def freeze_weights(resnet18):
        for name, param in resnet18.named_parameters():
            if not name.startswith('fc.'):
                param.requires_grad = False

    freeze_weights(resnet18)

We define callbacks for saving last epoch, best epoch and logging the results.

.. code-block:: python

    callbacks = [
        # Save the latest weights to be able to resume the optimization at the end for more epochs.
        ModelCheckpoint('last_epoch.ckpt', temporary_filename='last_epoch.ckpt.tmp'),

        # Save the weights in a new file when the current model is better than all previous models.
        ModelCheckpoint('best_epoch_{epoch}.ckpt', monitor='val_acc', mode='max', save_best_only=True,
                        restore_best=True, verbose=True, temporary_filename='best_epoch.ckpt.tmp'),

        # Save the losses and accuracies for each epoch in a TSV.
        CSVLogger('log.tsv', separator='\t'),
    ]


Finally, we start the training and output its final test loss, accuracy, and micro F1-score.

.. Note:: The F1-score is quite similar to the accuracy since the dataset is very balanced.

.. code-block:: python

    optimizer = optim.SGD(resnet18.fc.parameters(), lr=learning_rate, weight_decay=0.001)
    loss_function = nn.CrossEntropyLoss()

    model = Model(resnet18, optimizer, loss_function, batch_metrics=['accuracy'], epoch_metrics=['f1'])

    model.to(device)

    model.fit_generator(train_loader, valid_loader, epochs=n_epoch, callbacks=callbacks)

    test_loss, (test_acc, test_f1) = model.evaluate_generator(test_loader)
    print('Test:\n\tLoss: {}\n\tAccuracy: {}\n\tF1-score: {}'.format(test_loss, test_acc, test_f1))

.. code-block:: python

    logs = pd.read_csv('log.tsv', sep='\t')
    print(logs)

    best_epoch_idx = logs['val_acc'].idxmax()
    best_epoch = int(logs.loc[best_epoch_idx]['epoch'])
    print("Best epoch: %d" % best_epoch)

.. code-block:: python

    metrics = ['loss', 'val_loss']
    plt.plot(logs['epoch'], logs[metrics])
    plt.legend(metrics)
    plt.show()

.. code-block:: python

    metrics = ['acc', 'val_acc']
    plt.plot(logs['epoch'], logs[metrics])
    plt.legend(metrics)
    plt.show()


Since we have created checkpoints using callbacks, we can restore the best model from those checkpoints and test it.

.. code-block:: python

    resnet18 = models.resnet18(pretrained=False, num_classes=num_classes)

    model = Model(resnet18, 'sgd', 'cross_entropy', batch_metrics=['accuracy'], epoch_metrics=['f1'])

    model.to(device)

    model.load_weights('best_epoch_{epoch}.ckpt'.format(epoch=best_epoch))

    test_loss, (test_acc, test_f1) = model.evaluate_generator(test_loader)
    print('Test:\n\tLoss: {}\n\tAccuracy: {}\n\tF1-score: {}'.format(test_loss, test_acc, test_f1))