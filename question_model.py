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
seq_len = 25
h = 256
num_tags = 100
batch_size = 64

gpu = False

print("loading data")
start = time.time()
glove = np.load('glove.npy')

q1 = np.load('q1.npy').astype(np.int32)
q2 = np.load('q2.npy').astype(np.int32)

y = np.load('y.npy')

q1 = torch.from_numpy(q1)
q2 = torch.from_numpy(q2)
y = torch.from_numpy(y).float()

glove = torch.from_numpy(glove)

train_idx = int(np.floor(q1.size()[0] * 8 / 10))


q1_train = q1[ :train_idx]
q2_train = q2[ :train_idx]
Y_train = y[ :train_idx]

q1_test = q1[train_idx:]
q2_test = q2[train_idx:]
Y_test = y[train_idx:]

print(time.time() - start)
print("creating model")

load = False

if load:
    model = torch.load('question_model.p')
else:
    model = GRU_Model(h,glove,100,int(y.max()+1))
 
if gpu:
    model = torch.nn.DataParallel(model).cuda()

params = model.params

opt = optim.Adam(params, lr=0.001)
    
print("Beginning Traiing")

train_batches = q1_train.size()[0] // batch_size

test_batches = q1_test.size()[0] // batch_size


def train():
    
    avg_loss = 0
    model.train()
    start = time.time()


    for i in range(train_batches):
            
        data1, data2 ,target = getBatch(q1_train,q2_train,Y_train,i)

        data1 = Variable(data1.long())
        data2 = Variable(data2.long())
        target = Variable(target.float())
        
        if gpu:
            data1 = data1.cuda()
            data2 = data2.cuda()

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


    for i in range(test_batches):
         
        data1, data2 ,target = getBatch(q1_test,q2_test,Y_test,i)

        data1 = Variable(data1.long())
        data2 = Variable(data2.long())
        target = Variable(target.float())
        
        if gpu:
        
            data1 = data1.cuda()
            data2 = data2.cuda()
                                    

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




for i in range(5):
    train()
    test()
    torch.save(model,open('question_model.p','wb'))
