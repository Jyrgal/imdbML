#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""
# We first broke the rating into a classfication problem; 1,2,3,4,5.
# For classification, we used a LSTM network to be able to form relationships between
# different parts of text. However, as we experimented with this LSTM network,
# we noticed that without enough data, it was hard to reach a high accuracy without
# too many epochs so that overfitting doesnt occur. However, in order to maintain it's
# benefits, we added a GRU network alongside and combined the outputs. Preprocessing involved
# a simple elimination of any non-letter characters. Dropout was chosen at 0.5 to ensure
# that the network could be able to accurately choose a rating with only a select few nodes.
# For loss, CrossEntropyLoss was selected due to it's internal LogSoftmax and ability
# for multiclass selection.

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import torch.nn.functional as F
import re
import numpy as np
from random import randrange

device = torch.device('cuda:0')
###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    review = " ".join(sample)
    review = review.replace('\'', '')
    review = re.sub(r"</?\w+[^>]*>", '', review)
    review = re.sub(r"[^a-zA-Z']", ' ', review)
    final = review.split()
    final = [i for i in final if len(i) > 1]
    return final

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=50)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    datasetLabel = datasetLabel - 1
    return datasetLabel.long()

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    pred = netOutput.argmax(dim=1, keepdim=True)
    pred = torch.add(pred, 1)
    return pred.float()

###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()
        self.dropout_rate = 0.5
        self.input_size = 50
        self.hidden_size = 128
        self.hidden_size_linear = 200
        self.lstm = tnn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            bias=True,
            dropout=self.dropout_rate,
            num_layers=4,
            bidirectional=True)

        self.gru = tnn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=2,
            bidirectional=True,
            dropout=0.5
        )

        self.fc = tnn.Sequential(
            tnn.Linear(self.hidden_size*2, self.hidden_size_linear),
            #tnn.Linear(self.hidden_dim*3, self.hidden_dim_linear),
            tnn.BatchNorm1d(self.hidden_size_linear),
            tnn.ReLU(inplace=True),
            tnn.Linear(self.hidden_size_linear, 5)
        )

        self.dropout = tnn.Dropout(p=self.dropout_rate)


    def forward(self, input, length):
        initial_output, hid = self.gru(input)
        initial_output = initial_output[:, -1, :]
        packed_output, (hidden, cell) = self.lstm(input)
        output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        initial_output = self.dropout(initial_output)
        output = self.dropout(output)
        final = initial_output + output
        final = torch.div(final, 2)
        output = self.fc(self.dropout(final))
        return output


lossFunc = tnn.CrossEntropyLoss()
net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 1
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.001)
