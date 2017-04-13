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
import torchvision
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import time
import sys
from utils import *
from networks import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import hamming_loss

#import theano.sandbox.cuda.basic_ops as sbcuda

n = 20000
seq_len = 80
h = 512
num_tags = 100
batch_size = 64

gpu = False

print("loading data")
start = time.time()
glove = np.load('glove.npy')

features = np.load('features.npy')
y = np.load('y.npy')

features = torch.from_numpy(features)
y = torch.from_numpy(y).float()

glove = torch.from_numpy(glove)

train_idx = int(np.floor(features.size()[0] * 8 / 10))

features_train =features[ :train_idx] 
Y_train = y[ :train_idx]

features_test = features[ train_idx: ]
Y_test = y[train_idx:]

print(time.time() - start)
print("creating model")

load = True

if load:
    model = torch.load('artist_model.p')
else:
    model = GRU_Model(h,glove,100,int(y.max()+1))

params = model.params

opt = optim.Adam(params, lr=0.001)
    
print("Beginning Traiing")


def train():
    
    avg_loss = 0
    model.train()
    start = time.time()

    X1_train, X2_train, y_train , length = newDataset(features_train.numpy(), 
                                                      Y_train.numpy())

    for i in range(length):
            
        data1, data2 ,target = getBatch(X1_train,X2_train,y_train,i)

        data1 = Variable(data1.long())
        data2 = Variable(data2.long())
        target = Variable(target.float())
        
        opt.zero_grad()
        
        loss = model.cost(data1,data2,target)
        loss.backward()

        opt.step()

        avg_loss += loss        

    print("TRAINING loss: ", (avg_loss / length ).data[0] , ' time: ' , time.time() - start)

        
                
def test():
    
    avg_loss = 0
    model.eval()
    start = time.time()
    prob_same = []
    prob_diff = []

    X1_test, X2_test, y_test , length = newDataset(features_test.numpy(),
                                                   Y_test.numpy())


    for i in range(length):
         
        data1, data2 ,target = getBatch(X1_test,X2_test,y_test,i)

        data1 = Variable(data1.long())
        data2 = Variable(data2.long())
        target = Variable(target.float())
        
        loss = model.cost(data1,data2,target)
        
        p_same,p_diff  = model.predict_prob(data1,data2,target)

        prob_diff.append(p_diff)
        prob_same.append((p_same))

        avg_loss += loss        


    prob_same = np.concatenate(prob_same,axis=0)
    prob_diff = np.concatenate(prob_diff,axis=0)

    acc_same = (prob_same > .5).mean()
    acc_diff = (prob_diff > .5).mean()

    print("TESTING loss: ", (avg_loss / length ).data[0] , ' time: ' , time.time() - start)
    print("Prob Same: "  ,prob_same.mean(), " Acc Same " , acc_same , "  Prob Diff: " , prob_diff.mean() , " Acc Diff:  " , acc_diff )



def accuracy():

    #representation of all training points
    num_batches = features_train.shape[0] // 64
    #u1 = np.zeros( ( num_batches * 64 , 100) )

    u1 = Variable(torch.ones(num_batches * 64 , 100))

    for i in range(num_batches):

        x = Variable(torch.from_numpy(features_train[ i*64: (i+1)*64 ]))

        #u1[ i*64: (i+1)*64 ] = model.forward(x).data.numpy()
        u1[i * 64: (i + 1) * 64] = model.forward(x)


    y_hat = np.zeros(features_test.shape[0])

    for j in range(features_test.shape[0]):

        x = Variable(torch.from_numpy(features_test[ j ]))

        y_hat[j] = model.predict( u1, x, Y_train)

    #representation of all testing points
    #num_batches = features_test.shape[0] // 64
    #u2 = np.zeros( ( num_batches * 64 , 100) )

    #for i in range(num_batches):

    #   x = Variable(torch.from_numpy(features_test[ i*64: (i+1)*64 ]))
    #   u2[ i*64: (i+1)*64 ] = model.forward(x).data.numpy()


accuracy()

#for i in range(1):

#    train()
#    test()
#    torch.save(model,open('artist_model.p','wb'))
