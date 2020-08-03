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

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import torch.nn.functional as F
import re
import numpy as np
# import sklearn

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    review = " ".join(sample)
    #print(review)
    review = re.sub(r"</?\w+[^>]*>", '', review)
    review = re.sub(r"[^a-zA-Z']", ' ', review)
    final = review.split()
    final = [i for i in final if len(i) > 1]
    #print (final)

    return final

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    #batch is a list of numbers? representing each unique word
    #print(batch)
    #print(batch)
    # print(vocab.freqs.most_common(200))
    # temp_words = []
    # for word in vocab.freqs:
    #     if (vocab.freqs[word] > 400):
    #         #print(word, ": ", vocab.freqs[word])
    #         temp_words.append(word)
    #
    # temp_numerical = []
    # for i in temp_words:
    #     #print(vocab.stoi[i])
    #     temp_numerical.append(vocab.stoi[i])
    #
    # #print(final_list)
    # final_list = []
    # #print(temp_numerical)
    # #final_list = [i for i in batch if i not in temp_numerical]
    # for curr in batch:
    #     temp_list = [i for i in curr if i in temp_numerical]
    #     final_list.append(temp_list)
    # print(final_list)
    # print(batch)
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
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    datasetLabel = torch.true_divide(datasetLabel,5)
    #print("convertLabel: ", datasetLabel)
    return datasetLabel

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    #print("convertNetOutput: ", netOutput)
    #print(netOutput)
    #print(netOutput.flatten())
    # return torch.sigmoid(netOutput)
    return netOutput

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
        self.dropout_prob = 0.5
        self.input_dim = 50
        self.hidden_dim = 170
        self.lstm = tnn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bias=True,
            dropout=self.dropout_prob,
            num_layers=4,
            bidirectional=True)
        self.fc = tnn.Linear(
            in_features=self.hidden_dim*2,
            out_features=1)
        self.dropout = tnn.Dropout(p=self.dropout_prob)

    def forward(self, input, length):
        batchSize, _, _ = input.size()
        lstm_out, (hn, cn) = self.lstm(input)
        #print("length: ", length)
        hidden = self.dropout(torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1))
        out = self.fc(hidden.squeeze(0)).view(batchSize, -1)[:, -1]
        #print("output: ", out)
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

lossFunc = tnn.MSELoss()

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
optimiser = toptim.Adam(net.parameters(), lr=0.001)
