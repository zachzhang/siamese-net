
from __future__ import print_function
import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from networks import *

import time
from utils import *

'''
Prep Quora data for training
'''

n = 20000
seq_len = 25

dtype = torch.cuda.FloatTensor

# load all lyric data into pandas dataframe
df = pd.read_csv('/home/zz1409/Quora/train.csv')
df.fillna(' ' , inplace=True)


# Build vocabulary and tokenizer
vect = CountVectorizer(max_features=n)
vect.fit(pd.concat([df['question1'] , df['question2']]))
vocab = vect.vocabulary_
tok = vect.build_analyzer()

# Load glove vectors for word embedding
vocab, glove = load_glove(vocab)

# Convert text to sequence input
q1 = df['question1'].apply(lambda x: sent2seq(x, vocab, tok, seq_len))
q1 = np.array(list(q1),dtype=np.uint16)

q2 = df['question2'].apply(lambda x: sent2seq(x, vocab, tok, seq_len))
q2 = np.array(list(q2),dtype=np.uint16)

y = df['is_duplicate'].values

shuffle= np.random.permutation(q1.shape[0])

np.save('q1.npy',q1[shuffle])
np.save('q2.npy',q2[shuffle])
np.save('y.npy',y[shuffle])
np.save('glove.npy',glove)


