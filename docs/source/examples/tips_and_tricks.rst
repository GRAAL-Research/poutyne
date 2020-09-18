.. role:: hidden
    :class: hidden-section

.. _tips_and_tricks:

Tips and Tricks
*************************

.. note::
    - See the notebook `here <https://github.com/GRAAL-Research/poutyne/blob/master/examples/tips_and_tricks.ipynb>`_
    - Run in `Google Colab <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/tips_and_tricks.ipynb>`_

Poutyne also over a variety of tools for fine-tuning the information generated during the training, such as colouring the training update message, a progress bar, multi-GPUs, user callbacks interface and a user naming interface for the metrics' names.

We will explore those tools using a different problem than the one presented in :ref:`intro`

Let's import all the needed packages.

.. code-block:: python

    import os
    import pickle

    import fasttext
    import fasttext.util
    import requests
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import roc_auc_score
    from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
    from torch.utils.data import DataLoader

    from poutyne import set_seeds, Model, ModelCheckpoint, CSVLogger, Callback, SKLearnMetrics


Also, we need to set Pythons's, NumPy's and PyTorch's seeds by using Poutyne function so that our training is (almost) reproducible.

.. code-block:: python

    set_seeds(42)


Train a Recurrent Neural Network (RNN)
======================================

In this notebook, we train an RNN, or more precisely, an LSTM, to predict the sequence of tags associated with a given address, known as parsing address.

This task consists of detecting, by tagging, the different parts of an address such as the civic number, the street name or the postal code (or zip code). The following figure shows an example of such a tagging.

.. image:: /_static/img/address_parsing.png

Since addresses are written in a predetermined sequence, RNN is the best way to crack this problem. For our architecture, we will use two components, an RNN and a fully-connected layer.

Now, let's set our training constants. We first have the CUDA device used for training if one is present. Second, we set the batch size (i.e. the number of elements to see before updating the model) and the learning rate for the optimizer.

.. code-block:: python

    cuda_device = 0
    device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

    batch_size = 32
    lr = 0.1


RNN
---
For the first component, instead of using a vanilla RNN, we use a variant of it, known as a long short-term memory (LSTM) (to learn more about `LSTM <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_. For now, we use a single-layer unidirectional LSTM.

Also, since our data is textual, we will use the well-known word embeddings to encode the textual information. The LSTM input and hidden state dimensions will be of the same size. This size corresponds to the word embeddings dimension, which in our case will be the `French pre trained <https://fasttext.cc/docs/en/crawl-vectors.html>`_ fastText embeddings of dimension ``300``.

.. Note:: See this `discussion <https://discuss.pytorch.org/t/could-someone-explain-batch-first-true-in-lstm/15402>`_ for the explanation why we use the ``batch_first`` argument.

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

We use this layer to map the representation of the LSTM (``300``) to the tag space (8, the number of tags) and predict the most likely tag using a softmax.

.. code-block:: python

    input_dim = dimension # the output of the LSTM
    tag_dimension = 8

    fully_connected_network = nn.Linear(input_dim, tag_dimension)

The Dataset
-----------

Now let's download our dataset; it's already split into a train, valid and test set using the following.

.. code-block:: python

    def download_data(saving_dir, data_type):
    """
    Function to download the dataset using data_type to specify if we want the train, valid or test.
    """
        root_url = "https://graal-research.github.io/poutyne-external-assets/tips_and_tricks_assets/{}.p"

        url = root_url.format(data_type)
        r = requests.get(url)
        os.makedirs(saving_dir, exist_ok=True)

        open(os.path.join(saving_dir, f"{data_type}.p"), 'wb').write(r.content)

    download_data('./data/', "train")
    download_data('./data/', "valid")
    download_data('./data/', "test")


Now let's load in memory the data.

.. code-block:: python

    train_data = pickle.load(open("./data/train.p", "rb"))  # 80,000 examples
    valid_data = pickle.load(open("./data/valid.p", "rb"))  # 20,000 examples
    test_data = pickle.load(open("./data/test.p", "rb"))  # 30,000 examples

If we take a look at the training dataset, it's a list of ``80,000`` tuples where the first element is the full address, and the second element is a list of the tag (the ground truth).

.. code-block:: python

    train_data[0:2]

Here a snapshot of the output

.. image:: /_static/img/data_snapshot.png

Since the address is a text, we need to *convert* it into categorical value, such as word embeddings, for that we will use a vectorizer. This embedding vectorizer will be able to extract for every word embedding value.

.. code-block:: python

    class EmbeddingVectorizer:
        def __init__(self):
            """
            Embedding vectorizer
            """

            fasttext.util.download_model('fr', if_exists='ignore')
            self.embedding_model = fasttext.load_model("./cc.fr.``300``.bin")

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

Now, since all the addresses are not of the same size, it is impossible to batch them together since all elements of a tensor must have the same lengths. But there is a trick, padding!

The idea is simple. We add *empty* tokens at the end of each sequence up to the longest one in a batch. For the word vectors, we add vectors of 0 as padding. For the tag indices, we pad with -100s. We do so because of the :class:`~torch.nn.CrossEntropyLoss`, the accuracy metric and the :class:`~poutyne.F1` metric all ignore targets with values of ``-100``.

To do this padding, we use the ``collate_fn`` argument of the PyTorch :class:`~torch.utils.data.DataLoader` and on running time, that process will be done. One thing to take into account, since we pad the sequence, we need each sequence's lengths to unpad them in the forward pass. That way, we can pad and pack the sequence to minimize the training time (read `this good explanation <https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch>`_ of why we pad and pack sequences).

.. code-block:: python

    def pad_collate_fn(batch):
        """
        The collate_fn that can add padding to the sequences so all can have
        the same length as the longest one.

        Args:
            batch (List[List, List]): The batch data, where the first element
            of the tuple are the word idx and the second element are the target
            label.

        Returns:
            A tuple (x, y). The element x is a tuple containing (1) a tensor of padded
            word vectors and (2) their respective lengths of the sequences. The element
            y is a tensor of padded tag indices. The word vectors are padded with vectors
            of 0s and the tag indices are padded with -100s. Padding with -100 is done
            because the cross-entropy loss, the accuracy metric and the F1 metric ignores
            the targets with values -100.
        """

        # This gets us two lists of tensors and a list of integer.
        # Each tensor in the first list is a sequence of word vectors.
        # Each tensor in the second list is a sequence of tag indices.
        # The list of integer consist of the lengths of the sequences in order.
        sequences_vectors, sequences_labels, lengths = zip(*[
            (torch.FloatTensor(seq_vectors), torch.LongTensor(labels), len(seq_vectors))
            for (seq_vectors, labels) in sorted(batch, key=lambda x: len(x[0]), reverse=True)
        ])

        lengths = torch.LongTensor(lengths)

        padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=0)

        padded_sequences_labels = pad_sequence(sequences_labels, batch_first=True, padding_value=-100)

        return (padded_sequences_vectors, lengths), padded_sequences_labels


.. code-block:: python

    train_loader = DataLoader(train_data_vectorize, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    valid_loader = DataLoader(valid_data_vectorize, batch_size=batch_size, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_data_vectorize, batch_size=batch_size, collate_fn=pad_collate_fn)

Full Network
^^^^^^^^^^^^

Now, since we have packed the sequence, we cannot use the PyTorch :class:`~torch.nn.Sequential` constructor to define our model, so we will define the forward pass for it to unpack the sequences (again, read `this good explanation <https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch>`_ of why we pad and pack sequences).

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
            total_length = padded_sequences_vectors.shape[1]

            pack_padded_sequences_vectors = pack_padded_sequence(padded_sequences_vectors, lengths, batch_first=True)

            lstm_out, self.hidden_state = self.lstm_network(pack_padded_sequences_vectors)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=total_length)

            tag_space = self.fully_connected_network(lstm_out)
            return tag_space.transpose(-1, 1) # we need to transpose since it's a sequence

    full_network = FullNetWork(lstm_network, fully_connected_network)

Summary
-------

So we have created an LSTM network (``lstm_network``), a fully connected network (``fully_connected_network``), those two components are used in the full network. This full network used padded, packed sequences (defined in the forward pass), so we created the ``pad_collate_fn`` function to process the needed work. The DataLoader will conduct that process. Finally, when we load the data, this will be done using the vectorizer, so the address will be represented using word embeddings. Also, the address components will be converted into categorical value (from 0 to 7).

Now that we have all the components for the network let's define our SGD optimizer.

.. code-block:: python

    optimizer = optim.SGD(full_network.parameters(), lr)

Poutyne Callbacks
=================

One nice feature of Poutyne is :class:`callbacks <poutyne.Callback>`. Callbacks allow doing actions during the training of the neural network. In the following example, we use three callbacks. One that saves the latest weights in a file to be able to continue the optimization at the end of training if more epochs are needed. Another one that saves the best weights according to the performance on the validation dataset. Finally, another one that saves the displayed logs into a TSV file.

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

.. _making_your_own_callback:

Making Your own Callback
========================

While Poutyne provides a great number of :class:`predefined callbacks <poutyne.Callback>`, it is sometimes useful to make your own callback.

In the following example, we want to see the effect of temperature on the optimization of our neural network. To do so, we either increase or decrease the temperature during the optimization. As one can see in the result, temperature either as no effect or has a detrimental effect on the performance of the neural network. This is so because the temperature has for effect to artificially changing the learning rates. Since we have found the right learning rate, increasing or decreasing, it shows no improvement on the results.

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
            self.celoss = nn.CrossEntropyLoss()

        def forward(self, y_pred, y_true):
            y_pred = y_pred / self.temperature
            return self.celoss(y_pred, y_true)


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

Now let's test our training loop for one epoch using the accuracy as the batch metric.

.. code-block:: python

    model = Model(full_network, optimizer, loss_function, batch_metrics=['accuracy'])
    model.to(device)
    model.fit_generator(train_loader,
                        valid_loader,
                        epochs=1,
                        callbacks=callbacks)

Coloring
========

Also, Poutyne use by default a coloring template of the training step when the package ``colorama`` is installed.
One could either remove the coloring (``progress_options=dict(coloring=False)``) or set a different coloring template using the fields:
``text_color``, ``ratio_color``, ``metric_value_color``, ``time_color`` and ``progress_bar_color``.
If a field is not specified, the default colour will be used.

Here an example where we set the ``text_color`` to MAGENTA and the ``ratio_color`` to BLUE.

.. code-block:: python

    model.fit_generator(train_loader,
                        valid_loader,
                        epochs=1,
                        callbacks=callbacks,
                        progress_options=dict(coloring={"text_color": "MAGENTA", "ratio_color":"BLUE"}))


Epoch metrics
=============

It's also possible to used epoch metrics such as :class:`~poutyne.F1`. You could also define your own epoch metric using the :class:`~poutyne.EpochMetric` interface.

.. code-block:: python

    model = Model(full_network,
                  optimizer,
                  loss_function,
                  batch_metrics=['accuracy'],
                  epoch_metrics=['f1'])
    model.to(device)
    model.fit_generator(train_loader,
                        valid_loader,
                        epochs=1,
                        callbacks=callbacks)


Furthermore, you could also use the :class:`~poutyne.SKLearnMetrics` wrapper to wrap a Scikit-learn metric as an epoch metric. Below, we show how to compute the AUC ROC using the :class:`~poutyne.SKLearnMetrics` class. We have to inherit the class so that the data is passed into the right format for the scikit-learn ``roc_auc_score`` function.

.. code-block:: python

    class FlattenSKLearnMetrics(SKLearnMetrics):
        def forward(self, y_pred, y_true):
            y_pred = y_pred.softmax(1)
            y_pred = y_pred.transpose(2, 1).flatten(0, 1)
            y_true = y_true.flatten()
            return super().forward(y_pred, y_true)

    roc_epoch_metric = FlattenSKLearnMetrics(roc_auc_score,
                                             kwargs=dict(multi_class='ovr', average='macro'))
    model = Model(full_network,
                  optimizer,
                  loss_function,
                  batch_metrics=['accuracy'],
                  epoch_metrics=['f1', roc_epoch_metric])
    model.to(device)
    model.fit_generator(train_loader,
                        valid_loader,
                        epochs=1,
                        callbacks=callbacks)


Metric naming
=============

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
==========

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
