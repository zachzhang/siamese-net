
# coding: utf-8

# In[71]:

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

n = 20000
seq_len = 80
h = 128
num_tags = 100
batch_size = 64


dtype = torch.cuda.FloatTensor

# load all lyric data into pandas dataframe
df = pd.read_csv('/Users/Hadoop/Desktop/LyricScraper/SongTagger/lyric_data.csv', index_col=0)

# Sometimes the API returns an error message rather than actual lyrics. This removes it
bad_song = df['lyrics'].value_counts().index[0]
df[df['lyrics'] == bad_song] = ''

# only take the ones that we have data for
df.fillna('', inplace=True)
df = df[df['lyrics'] != '']
df = df.drop_duplicates(['artist','song'])

artists = df['artist'].value_counts()
a = pd.DataFrame(artists[(artists >4)].index)
a['id'] = a.index
a.columns = ['artist','label']
df = pd.merge(df,a,how='right',on='artist')
df = df.reindex(np.random.permutation(df.index))

# Build vocabulary and tokenizer
vect = CountVectorizer(max_features=n, stop_words='english')
vect.fit(df['lyrics'])
vocab = vect.vocabulary_
tok = vect.build_analyzer()

# Load glove vectors for word embedding
vocab, glove = load_glove(vocab)

# Convert text to sequence input
features = df['lyrics'].apply(lambda x: sent2seq(x, vocab, tok, seq_len))
features = np.array(list(features))

y = df['label'].values

np.save('features.npy',features)
np.save('y.npy',y)
np.save('glove.npy',glove)


# In[77]:



