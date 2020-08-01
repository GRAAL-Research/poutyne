.. role:: hidden
    :class: hidden-section

Examples
###################################

Here a list of all the examples materials for Poutyne.

.. _intro:

Introduction to PyTorch and Poutyne
***********************************

In this example, we train a simple fully-connected network and a simple convolutional network on MNIST. First, we train it by coding our own training loop as the PyTorch library expects of us to. Then, we use Poutyne to simplify our code.

.. code-block:: python

    # Import the package needed.
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data.dataset import Subset
    from torchvision import transforms, utils
    from torchvision.datasets.mnist import MNIST

    from poutyne import set_seeds
    from poutyne.framework import Model

.. code-block:: python

    # Set Pythons's, NumPy's and PyTorch's seeds so that our training are (almost) reproducible.
    set_seeds(42)

Basis of Training a Neural Network
==================================

In **stochastic gradient descent**, a **batch** of ``m`` examples are drawn from the train dataset. In the so-called forward pass, these examples are passed through the neural network and an average of their loss values is done. In the backward pass, the average loss is backpropagated through the network to compute the gradient of each parameter. In practice, the ``m`` examples of a batch are drawn without replacement. Thus, we define one **epoch** of training being the number of batches needed to loop through the entire training dataset.

In addition to the training dataset, a **validation dataset** is used to evaluate the neural network at the end of each epoch. This validation dataset can be used to select the best model during training and thus avoiding overfitting the training set. It also can have other uses such as selecting hyperparameters

Finally, a **test dataset** is used at the end to evaluate the final model.

Training constants
------------------

.. code-block:: python

    # Train on GPU if one is present
    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

    # The dataset is split 80/20 for the train and validation datasets respectively.
    train_split_percent = 0.8

    # The MNIST dataset has 10 classes
    num_classes = 10

    # Training hyperparameters
    batch_size = 32
    learning_rate = 0.1
    num_epochs = 5


Loading the MNIST dataset
-------------------------

The following loads the MNIST dataset and creates the PyTorch DataLoaders that split our datasets into batches. The train DataLoader shuffles the examples of the train dataset to draw the examples without replacement.

.. code-block:: python

    full_train_dataset = MNIST('./mnist/', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST('./mnist/', train=False, download=True, transform=transforms.ToTensor())

    num_data = len(full_train_dataset)
    indices = list(range(num_data))
    np.random.shuffle(indices)

    split = math.floor(train_split_percent * num_data)

    train_indices = indices[:split]
    train_dataset = Subset(full_train_dataset, train_indices)

    valid_indices = indices[split:]
    valid_dataset = Subset(full_train_dataset, valid_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    loaders = train_loader, valid_loader, test_loader


Let's take a look at some examples of the dataset.

.. code-block:: python

    # Get the first batch in our train DataLoader and
    # format it in grid.
    inputs = next(iter(train_loader))[0]
    input_grid = utils.make_grid(inputs)

    # Plot the images.
    fig = plt.figure(figsize=(10, 10))
    inp = input_grid.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.show()



Neural Network Architectures
----------------------------

We train a fully-connected neural network and a convolutional neural network with approximately the same number of parameters.

Fully-connected Network
^^^^^^^^^^^^^^^^^^^^^^^

In short, the fully-connected network follows this architecture: ``Input -> [Linear -> ReLU]*3 -> Linear``. The following table shows it in details:

.. list-table::
        :header-rows: 1

        *   - Layer Type
            - Output size
            - # of Parameters
        *   - Input
            - 1x28x28
            - 0
        *   - Flatten
            - 1\*28\*28
            - 0
        *   - **Linear with 256 neurons**
            - 256
            - 28\*28\*256 + 256 = 200,960
        *   - ReLU
            - \*
            - 0
        *   - **Linear with 128 neurons**
            - 128
            - 256\*128 + 128 = 32,896
        *   - ReLU
            - \*
            - 0
        *   - **Linear with 64 neurons**
            - 64
            - 128\*64 + 64 = 8,256
        *   - ReLU
            - \*
            - 0
        *   - **Linear with 10 neurons**
            - 10
            - 64\*10 + 10 = 650

Total # of parameters of the fully-connected network: 242,762

Convolutional Network
^^^^^^^^^^^^^^^^^^^^^

The convolutional neural network architecture starts with some convolution and max-pooling layers. These are then followed by fully-connected layers. We calculate the total number of parameters that the network needs. In short, the convolutional network follows this architecture: ``Input -> [Conv -> ReLU -> MaxPool]*2 -> Dropout -> Linear -> ReLU -> Dropout -> Linear``. The following table shows it in details:

.. list-table::
        :header-rows: 1

        *   - Layer Type
            - Output Size
            - # of Parameters
        *   - Input
            - 1x28x28
            - 0
        *   - **Conv with 16 3x3 filters with padding of 1**
            - 16x28x28
            - 16\*1\*3\*3 + 16 = 160
        *   - ReLU
            - 16x28x28
            - 0
        *   - MaxPool 2x2
            - 16x14x14
            - 0
        *   - **Conv with 32 3x3 filters with padding of 1**
            - 32x14x14
            - 32\*16\*3\*3 + 32 = 4,640
        *   - ReLU
            - 32x14x14
            - 0
        *   - MaxPool 2x2
            - 32x7x7
            - 0
        *   - Dropout of 0.25
            - 32x7x7
            - 0
        *   - Flatten
            - 32\*7\*7
            - 0
        *   - **Linear with 128 neurons**
            - 128
            - 32\*7\*7\*128 + 128 = 200,832
        *   - ReLU
            - 128
            - 0
        *   - Dropout of 0.5
            - 128
            - 0
        *   - **Linear with 10 neurons**
            - 10
            - 128\*10 + 10 = 1290

Total # of parameters of the convolutional network: 206,922

.. code-block:: python

    def create_fully_connected_network():
        """
        This function returns the fully-connected network layed out above.
        """
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def create_convolutional_network():
        """
        This function returns the convolutional network layed out above.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )



Training the PyTorch way
========================

That is, doing your own training loop.

.. code-block:: python

    def pytorch_accuracy(y_pred, y_true):
        """
        Computes the accuracy for a batch of predictions

        Args:
            y_pred (torch.Tensor): the logit predictions of the neural network.
            y_true (torch.Tensor): the ground truths.

        Returns:
            The average accuracy of the batch.
        """
        y_pred = y_pred.argmax(1)
        return (y_pred == y_true).float().mean() * 100

    def pytorch_train_one_epoch(pytorch_network, optimizer, loss_function):
        """
        Trains the neural network for one epoch on the train DataLoader.

        Args:
            pytorch_network (torch.nn.Module): The neural network to train.
            optimizer (torch.optim.Optimizer): The optimizer of the neural network
            loss_function: The loss function.

        Returns:
            A tuple (loss, accuracy) corresponding to an average of the losses and
            an average of the accuracy, respectively, on the train DataLoader.
        """
        pytorch_network.train(True)
        with torch.enable_grad():
            loss_sum = 0.
            acc_sum = 0.
            example_count = 0
            for (x, y) in train_loader:
                # Transfer batch on GPU if needed.
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                y_pred = pytorch_network(x)

                loss = loss_function(y_pred, y)

                loss.backward()

                optimizer.step()

                # Since the loss and accuracy are averages for the batch, we multiply
                # them by the the number of examples so that we can do the right
                # averages at the end of the epoch.
                loss_sum += float(loss) * len(x)
                acc_sum += float(pytorch_accuracy(y_pred, y)) * len(x)
                example_count += len(x)

        avg_loss = loss_sum / example_count
        avg_acc = acc_sum / example_count
        return avg_loss, avg_acc

    def pytorch_test(pytorch_network, loader, loss_function):
        """
        Tests the neural network on a DataLoader.

        Args:
            pytorch_network (torch.nn.Module): The neural network to test.
            loader (torch.utils.data.DataLoader): The DataLoader to test on.
            loss_function: The loss function.

        Returns:
            A tuple (loss, accuracy) corresponding to an average of the losses and
            an average of the accuracy, respectively, on the DataLoader.
        """
        pytorch_network.eval()
        with torch.no_grad():
            loss_sum = 0.
            acc_sum = 0.
            example_count = 0
            for (x, y) in loader:
                # Transfer batch on GPU if needed.
                x = x.to(device)
                y = y.to(device)

                y_pred = pytorch_network(x)
                loss = loss_function(y_pred, y)

                # Since the loss and accuracy are averages for the batch, we multiply
                # them by the the number of examples so that we can do the right
                # averages at the end of the test.
                loss_sum += float(loss) * len(x)
                acc_sum += float(pytorch_accuracy(y_pred, y)) * len(x)
                example_count += len(x)

        avg_loss = loss_sum / example_count
        avg_acc = acc_sum / example_count
        return avg_loss, avg_acc


    def pytorch_train(pytorch_network):
        """
        This function transfers the neural network to the right device,
        trains it for a certain number of epochs, tests at each epoch on
        the validation set and outputs the results on the test set at the
        end of training.

        Args:
            pytorch_network (torch.nn.Module): The neural network to train.

        Example:
            This function displays something like this:

            .. code-block:: python

                Epoch 1/5: loss: 0.5026924496193726, acc: 84.26666259765625, val_loss: 0.17258917854229608, val_acc: 94.75
                Epoch 2/5: loss: 0.13690324830015502, acc: 95.73332977294922, val_loss: 0.14024296019474666, val_acc: 95.68333435058594
                Epoch 3/5: loss: 0.08836929737279813, acc: 97.29582977294922, val_loss: 0.10380942322810491, val_acc: 96.66666412353516
                Epoch 4/5: loss: 0.06714504160980383, acc: 97.91874694824219, val_loss: 0.09626663728555043, val_acc: 97.18333435058594
                Epoch 5/5: loss: 0.05063822727650404, acc: 98.42708587646484, val_loss: 0.10017542181412378, val_acc: 96.95833587646484
                Test:
                    Loss: 0.09501855444908142
                    Accuracy: 97.12999725341797
        """
        print(pytorch_network)

        # Transfer weights on GPU if needed.
        pytorch_network.to(device)

        optimizer = optim.SGD(pytorch_network.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(1, num_epochs + 1):
            # Training the neural network via backpropagation
            train_loss, train_acc = pytorch_train_one_epoch(pytorch_network, optimizer, loss_function)

            # Validation at the end of the epoch
            valid_loss, valid_acc = pytorch_test(pytorch_network, valid_loader, loss_function)

            print("Epoch {}/{}: loss: {}, acc: {}, val_loss: {}, val_acc: {}".format(
                epoch, num_epochs, train_loss, train_acc, valid_loss, valid_acc
            ))

        # Test at the end of the training
        test_loss, test_acc = pytorch_test(pytorch_network, test_loader, loss_function)
        print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))

.. code-block:: python

    fc_net = create_fully_connected_network()
    pytorch_train(fc_net)

.. code-block:: python

    conv_net = create_convolutional_network()
    pytorch_train(conv_net)



Training the Poutyne way
========================

That is, only 8 lines of code with a better output.

.. code-block:: python

    def poutyne_train(pytorch_network):
        """
        This function creates a Poutyne Model (see https://poutyne.org/model.html), sends the
        Model on the specified device, and uses the `fit_generator` method to train the
        neural network. At the end, the `evaluate_generator` is used on  the test set.

        Args:
            pytorch_network (torch.nn.Module): The neural network to train.
        """
        print(pytorch_network)

        optimizer = optim.SGD(pytorch_network.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()

        # Poutyne Model
        model = Model(pytorch_network, optimizer, loss_function, batch_metrics=['accuracy'])

        # Send model on GPU
        model.to(device)

        # Train
        model.fit_generator(train_loader, valid_loader, epochs=num_epochs)

        # Test
        test_loss, test_acc = model.evaluate_generator(test_loader)
        print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))

.. code-block:: python

    fc_net = create_fully_connected_network()
    poutyne_train(fc_net)

.. code-block:: python

    conv_net = create_convolutional_network()
    poutyne_train(conv_net)





Poutyne's Tips and Tricks
*************************
Poutyne also over a variety of tools for fine-tuning the information generated during the training, such as colouring the training update message, a progress bar, multi-GPUs, user callbacks interface and a user naming interface for the metrics' names.

We will explore those tools using a different problem that the one presented in :ref:`intro`

.. code-block:: python

    import os
    import pickle

    import fasttext
    import fasttext.util
    import requests
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
    from torch.utils.data import DataLoader

    from poutyne.framework import Model, ModelCheckpoint, CSVLogger, Callback
    from poutyne.framework.metrics import F1
    from poutyne.framework.metrics import acc


Train a Recurrent Neural Network (RNN)
======================================

In this example, we train an RNN, or more precisely, an LSTM, to predict the sequence of tags associated with a given address, known as parsing address.

This task consists of detecting, by tagging, the different parts of an address such as the civic number, the street name or the postal code (or zip code). The following figure shows an example of such a tagging.

..      image:: /_static/img/address_parsing.png

Since addresses are written in a predetermined sequence, RNN is the best way to crack this problem. For our architecture, we will use two components, an RNN and a fully-connected layer.

Training Constants
------------------

.. code-block:: python

    batch_size = 32
    lr = 0.1

    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")



RNN
---
For the first components, instead of using a vanilla RNN, we will use a variant of it, know as a long short-term memory (LSTM) (to learn more about `LSTM <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_. For now, we will use a single layer unidirectional LSTM.

Also, since our data is textual, we will use the well-known word embeddings to encode the textual information. So the LSTM input and hidden state dimensions will be of the same size. This size corresponds to the word embeddings dimension, which in our case will be the `French pre trained <https://fasttext.cc/docs/en/crawl-vectors.html>`_ fastText embeddings of dimension 300.

.. Note:: See `this <https://discuss.pytorch.org/t/could-someone-explain-batch-first-true-in-lstm/15402>`_ for the explanation why we use the ``batch_first`` argument.

.. code-block:: python

    dimension = 300
    num_layer = 1
    bidirectional = False

    lstm_network = nn.LSTM(input_size=dimension,
                           hidden_size=dimension,
                           num_layers=num_layer,
                           bidirectional=bidirectional,
                           batch_first=True)


Fully-connected Layer
---------------------

We use this layer to map the representation of the LSTM (300) to the tag space (8, the number of tags) and predict the most likely tag using a softmax.

.. code-block:: python

    input_dim = dimension #the output of the LSTM
    tag_dimension = 8

    fully_connected_network = nn.Linear(input_dim, tag_dimension)

The Dataset
-----------

Now let's take a look at the dataset; it already split into a train, valid and test set using the following.

.. code-block:: python

    # Download the data from a directory

    #function to load the data from a repository
    def download_data(saving_dir, data_type):
        root_url = "https://graal-research.github.io/poutyne-external-assets/tips_and_tricks_assets/{}.p"

        url = root_url.format(data_type)
        r = requests.get(url)
        os.makedirs(saving_dir, exist_ok=True)

        open(os.path.join(saving_dir, f"{data_type}.p"), 'wb').write(r.content)

    download_data('./data/', "train")
    download_data('./data/', "valid")
    download_data('./data/', "test")

.. code-block:: python

    # load the data

    train_data = pickle.load(open("./data/train.p", "rb"))  # 80,000 examples
    valid_data = pickle.load(open("./data/valid.p", "rb"))  # 20,000 examples
    test_data = pickle.load(open("./data/test.p", "rb"))  # 30,000 examples

If we take a look at the training dataset, it's a list of 80,000 tuples where the first element is the full address, and the second element is a list of the tag (the ground truth).

.. code-block:: python

    train_data[0:2]

Since the address is a text, we need to *convert* it into categorical value, such as word embeddings, for that we will use a vectorizer. This embedding vectorizer will be able to extract for every word embedding value.

.. code-block:: python

    class EmbeddingVectorizer:
        def __init__(self):
            """
            Embedding vectorizer
            """

            fasttext.util.download_model('fr', if_exists='ignore')
            self.embedding_model = fasttext.load_model("./cc.fr.300.bin")

        def __call__(self, address):
            """
            Convert address to embedding vectors
            :param address: The address to convert
            :return: The embeddings vectors
            """
            embeddings = []
            for word in address.split():
                embeddings.append(self.embedding_model[word])
            return embeddings

    embedding_model = EmbeddingVectorizer()

We also need a vectorizer to convert the address tag (e.g. StreeNumber, StreetName) into categorical values. So we will use a Vectorizer class that can use the embedding vectorizer and convert the address tag.

.. code-block:: python

    class Vectorizer:
        def __init__(self, dataset, embedding_model):
            self.data = dataset
            self.embedding_model = embedding_model
            self.tags_set = {
                "StreetNumber": 0,
                "StreetName": 1,
                "Unit": 2,
                "Municipality": 3,
                "Province": 4,
                "PostalCode": 5,
                "Orientation": 6,
                "GeneralDelivery": 7
            }

        def __len__(self):
            # for the dataloader
            return len(self.data)

        def __getitem__(self, item):
            data = self.data[item]
            address = data[0]
            address_vector = self.embedding_model(address)

            tags = data[1]
            idx_tags = self._convert_tags_to_idx(tags)

            return address_vector, idx_tags

        def _convert_tags_to_idx(self, tags):
            idx_tags = []
            for tag in tags:
                idx_tags.append(self.tags_set[tag])
            return idx_tags


.. code-block:: python

    train_data_vectorize = Vectorizer(train_data, embedding_model)
    valid_data_vectorize = Vectorizer(valid_data, embedding_model)
    test_data_vectorize = Vectorizer(test_data, embedding_model)

DataLoader
^^^^^^^^^^

Now, since all the addresses are not of the same size, it is impossible to batch size them since all elements of a tensor must have the same lengths. But there a trick, padding!

The idea is simple. We will add *empty* tokens at the ends of a sequence up to the longest one in a batch. At the moment of evaluating the loss, that tokens will be skip using a mask value. That way, we can pad and pack the sequence to minimize the training time (read `here <https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch>`_ for a good explanation of why we pad and pack sequence).

For that, we will use the ``collate_fn`` of the PyTorch DataLoader, and on running time, that process will be done. We will create a class, that will save the padding value (`0`) and the mask value (``-100``) and use the ``__call__`` method to process the batch elements.

One time to take into account, since we have packed the sequence, we need the lengths of each sequence for the forward pass to unpack them.

.. code-block:: python

    class PadCollate:
        """
            A variant of collate_fn that pads the sequence to the longest sequence in the minibatch.
        """

        def __init__(self):
            self.pad_idx = 0
            self.mask_value = -100

        def _pad_collate_fn(self, batch):
            """
            **Args:**

                :batch - list of (List, List) where the first element of the tuple are the word idx and the second element
                are the target label.

            Returns:
                A list of the padded tensor sequence idx and the padded label tensor of size of the longest sequence length.

            """
            sequences_vectors, sequences_labels, lengths = zip(
                *[(torch.FloatTensor(seq_vectors), torch.LongTensor(labels), len(seq_vectors)) for (seq_vectors, labels)
                  in sorted(batch, key=lambda x: len(x[0]), reverse=True)])

            lengths = torch.LongTensor(lengths)

            padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=self.pad_idx)

            padded_sequences_labels = pad_sequence(sequences_labels, batch_first=True, padding_value=self.pad_idx)

            mask = self._mask_padding_sequences(lengths)
            masked_target = torch.where(mask, padded_sequences_labels,
                                        torch.ones_like(mask) * self.mask_value)

            # We also pass the mask for the F1 score sine it need a mask tensor to be compute
            return (padded_sequences_vectors, lengths), (masked_target, mask)

        def __call__(self, batch):
            return self._pad_collate_fn(batch)

        @staticmethod
        def _mask_padding_sequences(lengths):
            """
            Create a mask from the padding sequences lengths.

            Args:
                lengths: The lengths use to create the padded sequence.
            """

            max_len = lengths[0]
            mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
            return mask.bool()


.. code-block:: python

    train_loader = DataLoader(train_data_vectorize, batch_size=batch_size, shuffle=True, collate_fn=PadCollate())
    valid_loader = DataLoader(valid_data_vectorize, batch_size=batch_size, collate_fn=PadCollate())
    test_loader = DataLoader(test_data_vectorize, batch_size=batch_size, collate_fn=PadCollate())

Full Network
^^^^^^^^^^^^

Now, since we have packed the sequence, we cannot use the PyTorch ``nn.Sequential`` constructor to define our model, so we will define the forward pass for it to unpack the sequences.

.. code-block:: python

    class FullNetWork(nn.Module):
        def __init__(self, lstm_network, fully_connected_network):
            super().__init__()
            self.hidden_state = None

            self.lstm_network = lstm_network
            self.fully_connected_network = fully_connected_network

        def forward(self, padded_sequences_vectors, lengths):
            """
                Defines the computation performed at every call.
            """
            pack_padded_sequences_vectors = pack_padded_sequence(padded_sequences_vectors, lengths, batch_first=True)

            lstm_out, self.hidden_state = self.lstm_network(pack_padded_sequences_vectors)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

            tag_space = self.fully_connected_network(lstm_out)
            return tag_space.transpose(-1, 1) # we need to transpose since it's a sequence

    full_network = FullNetWork(lstm_network, fully_connected_network)

Summary
-------

So we have created an LSTM network (``lstm_network``), a fully connected network (``fully_connected_network``), those two components are used in the full network. This full network used padded, packed sequences (defined in the forward pass), so we created the ``PadCollate`` class to process the need work. The DataLoader will conduct that process. Finally, when we load the data, this will be done using the vectorizer, so the address will be represented using word embeddings. Also, the address components will be converted into categorical value (from 0 to 7).

The Training Loop
=================

Now that we have all the components for the network let's define our SGD optimizer.

.. code-block:: python

    optimizer = optim.SGD(full_network.parameters(), lr)

Poutyne Callbacks
-----------------

One nice feature of Poutyne is `callbacks <https://poutyne.org/callbacks.html>`_. Callbacks allow doing actions during the training of the neural network. In the following example, we use three callbacks. One that saves the latest weights in a file to be able to continue the optimization at the end of training if more epochs are needed. Another one that saves the best weights according to the performance on the validation dataset. Finally, another one that saves the displayed logs into a TSV file.

.. code-block:: python

    name_of_network = "lstm_unidirectional"

    callbacks = [
            # Save the latest weights to be able to continue the optimization at the end for more epochs.
            ModelCheckpoint(name_of_network + '_last_epoch.ckpt', temporary_filename='last_epoch.ckpt.tmp'),

            # Save the weights in a new file when the current model is better than all previous models.
            ModelCheckpoint(name_of_network + '_best_epoch_{epoch}.ckpt', monitor='val_accuracy', mode='max', save_best_only=True, restore_best=True, verbose=True, temporary_filename='best_epoch.ckpt.tmp'),

            # Save the losses and accuracies for each epoch in a TSV.
            CSVLogger(name_of_network + '_log.tsv', separator='\t'),
        ]

Making Your own Callback
------------------------

While Poutyne provides a great number of `predefined callbacks <https://poutyne.org/callbacks.html>`_, it is sometimes useful to make your own callback.

In the following example, we want to see the effect of temperature on the optimization of our neural network. To do so, we either increase or decrease the temperature during the optimization. As one can see in the result, temperature either as no effect or has a detrimental effect on the performance of the neural network. This is so because the temperature has for effect to artificially changing the learning rates. Since we have found the right learning rate, increasing or decreasing, it shows no improvement on the results.

.. Note:: Since we use a mask, y_true is a tuple where the first element is the ground truth and the second one is the mask.

.. code-block:: python

    class CrossEntropyLossWithTemperature(nn.Module):
        """
        This loss module is the cross-entropy loss function
        with temperature. It divides the logits by a temperature
        value before computing the cross-entropy loss.

        Args:
            initial_temperature (float): The initial value of the temperature.
        """

        def __init__(self, initial_temperature):
            super().__init__()
            self.temperature = initial_temperature
            self.celoss = nn.CrossEntropyLoss(ignore_index=-100)  # we use the same -100 ignore index

        def forward(self, y_pred, y_true):
            y_pred = y_pred / self.temperature
            # Since y_true is a tuple where y_true[1] is the mask
            return self.celoss(y_pred, y_true[0])


    class TemperatureCallback(Callback):
        """
        This callback multiply the loss temperature with a decay before
        each batch.

        Args:
            celoss_with_temp (CrossEntropyLossWithTemperature): the loss module.
            decay (float): The value of the temperature decay.
        """
        def __init__(self, celoss_with_temp, decay):
            super().__init__()
            self.celoss_with_temp = celoss_with_temp
            self.decay = decay

        def on_train_batch_begin(self, batch, logs):
            self.celoss_with_temp.temperature *= self.decay

So our loss function will be the cross-entropy with temperature with an initial temperature of ``0.1`` and a temperature decay of ``1.0008``.

.. code-block:: python

    loss_function = CrossEntropyLossWithTemperature(0.1)
    callbacks = callbacks + [TemperatureCallback(loss_function, 1.0008)]

Finally, as we saw early, ``y_true`` is a tuple, so we need to modify a little bit the accuracy.

.. code-block:: python

    def accuracy(y_pred, y_true, ignore_index=-100):
        """
        Wrapper function around the accuracy where the y is a tuple of (tag, mask).
        """

        # Since y_true[1] is the mask
        return acc(y_pred, y_true=y_true[0], ignore_index=ignore_index)

Now let's test our training loop for one epoch using the accuracy as the batch metric.

.. code-block:: python

    model = Model(full_network, optimizer, loss_function, batch_metrics=[accuracy])
    model.to(device)
    model.fit_generator(train_loader,
                        valid_loader,
                        epochs=1,
                        callbacks=callbacks)

Coloring
--------

Also, Poutyne use by default a coloring template of the training step when the package ``colorama`` is installed.
One could either remove the coloring (``color_log=False``) or set a different coloring template using the fields:
``text_color``, ``ratio_color``, ``metric_value_color``, ``time_color`` and ``progress_bar_color``.
If a field is not specified, the default colour will be used.

Here an example where we set the ``text_color`` to MAGENTA and the ``ratio_color`` to BLUE.

.. code-block:: python

    model.fit_generator(train_loader,
                        valid_loader,
                        epochs=1,
                        callbacks=callbacks,
                        coloring={"text_color": "MAGENTA", "ratio_color":"BLUE"})


Epoch metrics
-------------

It's also possible to used epoch metrics such as F1-score. You could also define your own epoch metric using the ``EpochMetric`` interface.

Furthermore, you could also use the ``SKLearnMetrics`` wrapper to wrap a Scikit-learn metric as an epoch metric.

.. code-block:: python

    model = Model(full_network,
                  optimizer,
                  loss_function,
                  batch_metrics=[accuracy],
                  epoch_metrics=[F1()])
    model.to(device)
    model.fit_generator(train_loader,
                        valid_loader,
                        epochs=1,
                        callbacks=callbacks)

Metric naming
-------------

It's also possible to name the metric using a tuple format ``(<metric name>, metric)``. That way, it's possible to use multiple times the same metric type (i.e. having micro and macro F1-score).

.. code-block:: python

    model = Model(full_network,
                  optimizer,
                  loss_function,
                  batch_metrics=[("My accuracy name", accuracy)],
                  epoch_metrics=[("My metric name", F1())])
    model.to(device)
    model.fit_generator(train_loader,
                        valid_loader,
                        epochs=1)

Multi-GPUs
----------

Finally, it's also possible to use multi-GPUs for your training either by specifying a list of devices or using the arg ``"all"`` to take them all.

.. Note:: Obviously, you need more than one GPUs for that option.

.. code-block:: python

    model = Model(full_network,
                  optimizer,
                  loss_function,
                  batch_metrics=[("My accuracy name", accuracy)],
                  epoch_metrics=[("My metric name", F1())])
    model.to("all")
    model.fit_generator(train_loader,
                        valid_loader,
                        epochs=1)


Interface of the ``policy`` module
**********************************

About the ``policy`` Module Interface
=====================================

The ``policy`` modules give you fine-grained control over the training process.
This example demonstrates how the ``policy`` module works and how you can create your own policies.

.. code-block:: python

    import matplotlib.pyplot as plt


Parameter Spaces and Phases
---------------------------

Parameter spaces like ``linspace`` and ``cosinespace`` are the basic building blocks.

.. code-block:: python

    from poutyne.framework import linspace, cosinespace


You can define the space and iterate over them:

.. code-block:: python

    space = linspace(1, 0, 3)
    for i in space:
        print(i)

.. code-block:: python

    space = cosinespace(1, 0, 5)
    for i in space:
        print(i)


You can use the space and create a phase with them:

.. code-block:: python

    from poutyne.framework import Phase

    phase = Phase(lr=linspace(0, 1, 3))

    # and iterate
    for d in phase:
        print(d)


You can also visualize your phase:

.. code-block:: python

    phase.plot("lr");


Phases can have multiple parameters:

.. code-block:: python

    phase = Phase(
        lr=linspace(0, 1, 10),
        momentum=cosinespace(.99, .9, 10),
    )

    phase.plot("lr");
    phase.plot("momentum")


Visualize Different Phases
--------------------------

.. code-block:: python

    steps = 100

    fig, ax = plt.subplots()
    # Constant value
    Phase(lr=linspace(.7, .7, steps)).plot(ax=ax)
    # Linear
    Phase(lr=linspace(0, 1, steps)).plot(ax=ax)
    # Cosine
    Phase(lr=cosinespace(1, 0, steps)).plot(ax=ax);


Visualize Multiple Parameters in One Phase
------------------------------------------

.. code-block:: python

    steps = 100
    phase = Phase(lr=linspace(1, 0.5, steps), momentum=cosinespace(.8, 1, steps))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    phase.plot("lr", ax=axes[0])
    phase.plot("momentum", ax=axes[1]);


Build Complex Policies From Basic Phases
========================================

You can build complex optimizer policies by chaining phases together:

.. code-block:: python

    from poutyne.framework import OptimizerPolicy

    policy = OptimizerPolicy([
        Phase(lr=linspace(0, 1, 100)),
        Phase(lr=cosinespace(1, 0, 200)),
        Phase(lr=linspace(0, .5, 100)),
        Phase(lr=linspace(.5, .1, 300)),
    ])

    policy.plot();


Use Already Defined Complex Policies
------------------------------------

It's easy to build your own policies, but Poutyne contains some pre-defined phases.

.. code-block:: python

    from poutyne.framework import sgdr_phases

    # build them manually
    policy = OptimizerPolicy([
        Phase(lr=cosinespace(1, 0, 200)),
        Phase(lr=cosinespace(1, 0, 400)),
        Phase(lr=cosinespace(1, 0, 800)),
    ])
    policy.plot()

    # or use the pre-defined one
    policy = OptimizerPolicy(sgdr_phases(base_cycle_length=200, cycles=3, cycle_mult=2))
    policy.plot();


Pre-defined ones are just a list phases:

.. code-block:: python

    sgdr_phases(base_cycle_length=200, cycles=3, cycle_mult=2)


Here is the one-cycle policy:

.. code-block:: python

    from poutyne.framework import one_cycle_phases

    tp = OptimizerPolicy(one_cycle_phases(steps=500))
    tp.plot("lr")
    tp.plot("momentum");


Train CIFAR with the ``policy`` module
**************************************


.. code-block:: python

    # Import
    import torch
    import torchvision.datasets as datasets
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from torchvision.models import resnet18
    import torch.nn as nn
    import torch.optim as optim
    from poutyne.framework import Model
    from poutyne.framework import OptimizerPolicy, one_cycle_phases


.. code-block:: python

    # Training constants
    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")


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


    BATCH_SIZE = 1024

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
    )


The model
=========

We'll train a simple resnet18 network.
This takes a while without GPU but is pretty quick with GPU.

.. code-block:: python

    def get_module():
        model = resnet18(pretrained=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 10)
        return model

.. code-block:: python

    epochs = 5


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

Transfer learning example
**************************

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

.. code-block:: python

    # Set Pythons's, NumPy's and PyTorch's seeds so that our training are (almost) reproducible.
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

.. code-block:: python

    # Training constants
    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

.. code-block:: python

    # Training hyperparameters
    batch_size = 32
    learning_rate = 0.1
    n_epoch = 30
    num_classes = 200


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


We load a pretrained ResNet-18 networks and replace the head with the number of neurons equal to our number of classes.

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
.. code-block:: python md
We define callbacks for saving last epoch, best epoch and logging the results

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

    test_loss, test_acc = model.evaluate_generator(test_loader)
    print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))

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

    test_loss, test_acc = model.evaluate_generator(test_loader)
    print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))