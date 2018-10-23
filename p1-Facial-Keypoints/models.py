## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


def hidden_init(layer):
    '''
    Provide fan in (the number of input units) of each hidden layer
    as the component of normalizer.

    :param
        layer: hidden layer
    :return
        (-lim, lim): tuple of min and max value for uniform distribution
    '''

    fan_in = layer.weight.data.size()[0]
    lim = 1/np.sqrt(fan_in)
    return -lim, lim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## TODO: Don't apply batch_normalization layer on the first and output layer, it may
        ## TODO: cause unexpected result.
        ## TODO: And set bias=False on layers with batch_normalization, for
        ## TODO: bias is already being taken into account in batch_normalization.
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # output (32, 220, 220)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False) # output (64, 216, 216)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.max2 = nn.MaxPool2d(5, 4)  # output (64, 53, 53)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=False) # output (128, 49, 49)
        self.bnorm3 = nn.BatchNorm2d(128)
        self.max3 = nn.MaxPool2d(5, 4) # output (128, 12, 12)

        self.fc1 = nn.Linear(128*12*12, 136*50, bias=False)
        self.fcBnorm1 = nn.BatchNorm1d(136*50)
        self.drop1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(136*50, 136*30, bias=False)
        self.fcBnorm2 = nn.BatchNorm1d(136*30)
        self.fc3 = nn.Linear(136*30, 136*10, bias=False)
        self.fcBnorm3 = nn.BatchNorm1d(136*10)
        self.fc4 = nn.Linear(136*10, 136)


        # Weight innitialization using Xavier innitialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                min_, max_ = hidden_init(m)
                m.weight.data.uniform_(min_, max_)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.bnorm2(self.conv2(x)))
        x = self.max2(x)
        x = F.relu(self.bnorm3(self.conv3(x)))
        x = self.max3(x)
        x = F.relu(self.fcBnorm1(self.fc1(x.view(x.size(0), -1))))
        x = self.drop1(x)
        x = F.relu(self.fcBnorm2(self.fc2(x)))
        x = F.relu(self.fcBnorm3(self.fc3(x)))
        logits = self.fc4(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return logits

