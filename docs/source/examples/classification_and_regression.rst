.. role:: hidden
    :class: hidden-section

Gender Classification and Eyes Location Detection: A Two Task Problem
*********************************************************************

.. note::

    - See the notebook `here <https://github.com/GRAAL-Research/poutyne/blob/master/examples/classification_and_regression.ipynb>`_
    - Run in `Google Colab <https://colab.research.google.com/github/GRAAL-Research/poutyne/blob/master/examples/classification_and_regression.ipynb>`_

In this example, we are going to implement a multi-task problem. We try to identify the gender of the people, as well as locating their eyes in the image. Hence, we have two different tasks: classification (to identify the gender) and regression (to find the location of the eyes). We are going to use a single network (A CNN) to perform both tasks, however, we will need to apply different loss functions, each proper to a specific task.

Let's import all the needed packages.

.. code-block:: python

    import math
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import wget
    import zipfile
    import cv2
    from natsort import natsorted
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.autograd import Variable
    import torchvision.datasets as datasets
    import torchvision.models as models
    import torchvision.transforms as tfms
    from poutyne import set_seeds, Model, ModelCheckpoint, CSVLogger, Experiment, StepLR
    from torch.utils.data import DataLoader, Subset, Dataset
    from torchvision.utils import make_grid

Training Constants
==================

.. code-block:: python

    num_epochs = 15
    learning_rate = 0.01
    batch_size = 32
    image_size = 224
    valid_split_percent = 0.1
    momentum = 0.5
    set_seeds(42)
    imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing 
    imagenet_std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('The running processor is...', device) 

CelebA Dataset
==============

We are going to use CelebA dataset for this experiment. The CelebA dataset  is a large-scale face attributes dataset which can be employed as the training and test sets for the following computer vision tasks: face attribute recognition, face detection, landmark (or facial part) localization, and face editing & synthesis.

Fetching data 
=============

The section below consists of a few lines of codes that help us download celebA dataset from a public web source and unzip them.

.. code-block:: python

    data_root = 'data/celeba'
    # Path to folder with the dataset
    dataset_folder = f'{data_root}/img_align_celeba'
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(dataset_folder, exist_ok=True)

    # URL for the CelebA dataset (aligned images, attributes, landmasrks)
    url = 'https://graal.ift.ulaval.ca/public/celeba/img_align_celeba.zip'
    attr_url = 'https://graal.ift.ulaval.ca/public/celeba/list_attr_celeba.txt'
    land_mark_url = 'https://graal.ift.ulaval.ca/public/celeba/list_landmarks_align_celeba.txt'

    # Path to download the dataset to
    download_path = f'{data_root}/img_align_celeba.zip'
    land_mark_path = f'{data_root}/list_landmarks_align_celeba.txt'
    attr_path = f'{data_root}/list_attr_celeba.txt'

    # Download the dataset from the source
    wget.download(url,download_path)
    wget.download(land_mark_url,land_mark_path)
    wget.download(attr_url,attr_path)

    # Path to folder with the dataset
    dataset_folder = f'{data_root}/img_align_celeba'
    os.makedirs(dataset_folder, exist_ok=True)

    # Unzip the downloaded file 
    with zipfile.ZipFile(download_path, 'r') as ziphandler:
        ziphandler.extractall(dataset_folder)
   
Create a custom dataset class
=============================

As we are going to implement a multi-task problem by a single CNN, we should provide the CNN with the ground truth in a proper way. Here, we have two different tasks: classification and regression. In the classification task, the goal is to identify the gender. The labels of the gender for each image are saved in the `list_attr_celeba.txt` file, in which 1 stands for male and -1 for female. Since we consider the loss of both tasks simultaneously, we scale all target values to the range of [0,1]. Hence, the gender labels will be changed as well, 1 for male and 0 for female. For the localization part, the coordinates of the eyes (Left and Right) are provided in the `list_landmarks_align_celeba.txt` file. In addition to scaling the number to the range of [0,1], we also need to rescale the coordinates to the image's new size (224,224).

.. code-block:: python

    class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        img_folder = data_root + '/img_align_celeba/img_align_celeba'
        image_names = os.listdir(img_folder)
        self.root_dir = img_folder
        self.data_root = data_root
        self.transform = transform 
        self.image_names = natsorted(image_names)

    def __len__(self): 
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        img = cv2.imread(img_path)
        w, h, _ = img.shape
        img = cv2.resize(img, (image_size, image_size))

        # Apply transformations to the image
        if self.transform:
        img = self.transform(img)

        img.requires_grad=True
        land_mark = open(f'{self.data_root}/list_landmarks_align_celeba.txt','r').readlines()[idx+2]
        land_mark_contents = land_mark.split(' ')
        land_mark_contents = [x for x in land_mark_contents if x]
        x_L, y_L ,x_R, y_R = int(land_mark_contents[1]), int(land_mark_contents[2]), int(land_mark_contents[3]), int(land_mark_contents[4])
        w_scale = image_size/w
        h_scale = image_size/h
        x_L, x_R = (x_L*h_scale/h), (x_R*h_scale/h)  #rescaling for the size of (224,224) and finaly to the range of [0,1]
        y_L, y_R = (y_L*w_scale/w), (y_R*w_scale/w)
        attr = open(f'{self.data_root}/list_attr_celeba.txt','r').readlines()[idx+2]
        attr_contents = attr.split(' ')
        attr_contents = [x for x in attr_contents if x]
        gender = attr_contents[21]
        gender = int((int(gender)+1)/2)
        return img, (torch.tensor(gender), torch.tensor([x_L, y_L, x_R, y_R], requires_grad=True),[w, h])

    transform=tfms.Compose([
        tfms.ToTensor(),
        tfms.Normalize(imagenet_mean, imagenet_std)
    ])

    celeba_dataset = CelebADataset(data_root, transform)
    celeba_dataloader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)
    full_dataset_length = len(celeba_dataset)
    indices = list(np.arange(full_dataset_length))
    np.random.shuffle(indices)
    train_indices = indices[math.floor(full_dataset_length * valid_split_percent):]
    valid_indices = indices[:math.floor(full_dataset_length * valid_split_percent)]
    train_dataset = Subset(celeba_dataset, train_indices)
    valid_dataset = Subset(celeba_dataset, valid_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

Here we can see how each dataset sample looks like:

.. code-block:: python

    print (train_dataset[0])

Here, we can see an example from the training dataset. It shows an image of a person, printing the gender and also showing the location of the eyes.

.. code-block:: python

    sample_number = 16
    image = train_dataset[sample_number][0]
    image = image.permute(1,2,0).detach().numpy()
    image_rgb = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb * imagenet_std + imagenet_mean
    Gender = 'male' if int(train_dataset[sample_number][1][0])==1 else 'female'
    print('Gender is: ', Gender)
    w, h = train_dataset[sample_number][1][2]
    (x1, y1) = train_dataset[sample_number][1][1][0:2]
    (x2, y2) = train_dataset[sample_number][1][1][2:4]
    x1, x2 = int(x1*h), int(x2*h)
    y1, y2 = int(y1*w), int(y2*w)
    image_rgb = cv2.drawMarker(image_rgb, (x1,y1), (0,255,0))
    image_rgb = cv2.drawMarker(image_rgb, (x2,y2), (0,255,0))
    image_rgb = np.clip(image_rgb , 0, 1)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

.. image:: /_static/img/classification_and_regression/dataset_sample.png

Network
=======

Below, we define a new class, named 'ClassifierLocalizer, which accepts a pre-trained CNN and changes its last fully connected layer to be proper for the two task problem. The new fully connected layer contains 6 neurons, 2 for the classification task (male or female) and 4 for the localization task (x and y for the left and right eyes). Moreover, to put the location results on the same scale as the class scores, we apply the sigmoid function to the neurons assigned for the localization task.

.. code-block:: python

    class ClassifierLocalizer(nn.Module):
        def __init__(self, model_name, num_classes=2):
            super(ClassifierLocalizer, self).__init__()
            self.num_classes = num_classes
            
            # create cnn model
            model = getattr(models, model_name)(pretrained=True)
            
            # remove fc layers and add a new fc layer
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 6) # classifier + localizer
            self.model = model
        
        def forward(self, x):
            x = self.model(x)                    # extract features from CNN
            scores = x[:, :self.num_classes]     # class scores
            coords = x[:, self.num_classes:]     # coordinates
            return [scores, torch.sigmoid(coords)]   # sigmoid output is in the range of [0, 1]

Regarding the complexity of the problem, the number of the samples in the training dataset, and the similarity of the training dataset to the ImageNet dataset, we may decide to freeze some of the layers. In our current example, based on the mentioned factors, we freeze just the last fully connected layer.

.. code-block:: python

    network = ClassifierLocalizer(model_name='resnet18')

    def freeze_weights(network):
        for name, param in network.named_parameters():
            if not name.startswith('fc.'):
                param.requires_grad = False

    freeze_weights(network)
    print(network)

.. code-block:: python

    network = ClassifierLocalizer(model_name='resnet18')  # network without freezing any layer.

Loss function
=============

As we discussed before, we have two different tasks in this example. These tasks need different loss functions; Cross-Entropy loss for the classification and Mean Square Error loss for the regression. Below, we define a new loss function class that sums both losses to considers them simultaneously. However, as the regression is relatively a harder task, we apply a higher weight to MSEloss.

.. code-block:: python

    class ClassificationRegressionLoss(nn.Module):
        def __init__(self):
            super(ClassificationRegressionLoss, self).__init__()
            self.ce_loss = nn.CrossEntropyLoss() # size_average=False
            self.mse_loss = nn.MSELoss()
            
        def forward(self, y_pred, y_true):
            loss_cls = self.ce_loss(y_pred[0], y_true[0]) # Cross Entropy Error (for classification)
            loss_reg = self.mse_loss(y_pred[1], y_true[ 1]) # Mean Squared Error (for landmarks)
            total_loss = loss_reg + loss_cls
            return total_loss

Training
========

.. code-block:: python

    optimizer = optim.Adam(network.parameters(), lr=0.0001, weight_decay=0)
    loss_function = ClassificationRegressionLoss()
    #Step_Learning_Rate = StepLR(step_size=2 , gamma=0.1, last_epoch=-1, verbose=False)
    exp = Experiment('./two_task_example', network, optimizer=optimizer, loss_function=loss_function, device="all")
    exp.train(train_dataloader, valid_dataloader, callbacks=callbacks, epochs=num_epochs)

Evaluation
==========

As you have also noticed from the training logs, we have achieved the best performance (considering the validation loss) at the 15th epoch. The weights of the network for the corresponding epoch have been automatically saved and we use these parameters to evaluate our algorithm visually. Hence,  we take advantage of evaluate function of Poutyne, and apply it to the validation dataset. It provides us the predictions as well as the ground-truth for comparison, in case of need.

.. code-block:: python

    model = Model(network, optimizer, loss_function, device=device)
    model.load_weights('./two_task_example/checkpoint_epoch_15.ckpt')
    loss, predictions, Ground_Truth = model.evaluate_generator(valid_dataloader, callbacks=callbacks, return_pred=True, return_ground_truth=True)


The ``callbacks`` feature also records the training logs. we can use this information to monitor and analyze the training process.

.. code-block:: python

    logs = pd.read_csv('./callbacks/log.tsv', sep='\t')
    print(logs)

.. image:: /_static/img/classification_and_regression/logs.png

.. code-block:: python

    train_loss = logs.loss
    valid_loss = logs.val_loss
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.legend(['train_loss','valid_loss'])
    plt.title('training and validation losses')
    plt.show()

.. image:: /_static/img/classification_and_regression/loss_diagram.png

We can also evaluate the performance of the trained network (a network with the best weights) on any dataset, as below:

.. code-block:: python

    exp.test(valid_dataloader)

Now let's evaluate the performance of the network visually.

.. code-block:: python

    sample_number = 10
    image = valid_dataset[sample_number][0]
    image = image.permute(1,2,0).detach().numpy()
    image_rgb = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb * imagenet_std + imagenet_mean
    Gender = 'male' if np.argmax(predictions[0][sample_number])==0 else 'female'
    print('Gender is: ', Gender)
    w, h = valid_dataset[sample_number][1][2]
    (x1, y1) = predictions[1][sample_number][0:2]
    (x2, y2) = predictions[1][sample_number][2:4]
    x1, x2 = int(x1*h), int(x2*h)
    y1, y2 = int(y1*w), int(y2*w)
    image_rgb = cv2.drawMarker(image_rgb, (x1,y1), (0,255,0))
    image_rgb = cv2.drawMarker(image_rgb, (x2,y2), (0,255,0))
    image_rgb = np.clip(image_rgb , 0, 1)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

.. image:: /_static/img/classification_and_regression/output_sample.png