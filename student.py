#!/usr/bin/env python3

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import torch.nn.functional as F
import re
# import numpy as np
# import sklearn


def preprocessing(sample):
    review = " ".join(sample)
    #print(review)
    review = re.sub(r"</?\w+[^>]*>", '', review)
    review = re.sub(r"[^a-zA-Z']", ' ', review)
    final = review.split()
    final = [i for i in final if len(i) > 1]
    print(final)
    """
    Called after tokenising but before numericalising.
    """

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """

    return batch

stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
             'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
             'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'got', 'it\'s', 'it.']
wordVectors = GloVe(name='6B', dim=50)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    print("convertLabel: ", datasetLabel)
    return datasetLabel

def convertNetOutput(netOutput):
    print("convertNetOutput: ", netOutput)
    return netOutput

###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):

    def __init__(self):
        super(network, self).__init__()
        self.dropout_prob = 0.5
        self.input_dim = 50
        self.hidden_dim = 170
        self.lstm = tnn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bias=True,
            dropout=self.dropout_prob,
            num_layers=2,
            bidirectional=True)
        self.fc = tnn.Linear(
            in_features=self.hidden_dim*2,
            out_features=1)
        self.dropout = tnn.Dropout(p=self.dropout_prob)

    def forward(self, input, length):
        batchSize, _, _ = input.size()
        lstm_out, (hn, cn) = self.lstm(input)
        hidden = self.dropout(torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1))
        out = self.fc(hidden.squeeze(0)).view(batchSize, -1)[:, -1]
        return out

# class loss(tnn.Module):
#     """
#     Class for creating a custom loss function, if desired.
#     You may remove/comment out this class if you are not using it.
#     """
#
#     def __init__(self):
#         super(loss, self).__init__()
#
#     def forward(self, output, target):
#         pass

lossFunc = tnn.BCEWithLogitsLoss()

net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
#lossFunc = loss()
###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 3
optimiser = toptim.SGD(net.parameters(), lr=0.01)
