
# coding: utf-8

# In[1]:

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

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]) , download=True ),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]), download=True ),
    batch_size=64, shuffle=True)


print("Creaing Dataset")

#train_loader = BatchLoader(np.expand_dims(train_loader.dataset.train_data.numpy() ,axis=1) / 255., train_loader.dataset.train_labels.numpy())
#test_loader = BatchLoader(np.expand_dims(test_loader.dataset.test_data.numpy(),axis=1) / 255. , test_loader.dataset.test_labels.numpy())



class SiameseNN(nn.Module):
    
    def __init__(self):
        super(SiameseNN, self).__init__()
    
        self.arc = [1,32 , 64, 64, 128]
        shapes = list(zip(self.arc[:-1], self.arc[1:]))

        kernels = [5,3,3,3]
        
        self.conv_layers = [ nn.Conv2d(s[0], s[1], kernel_size=k) for s,k in zip(shapes,kernels)  ]
        self.dropout = [nn.Dropout2d(.5) for k in kernels ] 
        self.bn = [nn.BatchNorm2d(size) for size in self.arc[1:] ]

        self.flat_dim = self.get_flat_dim()
        
        self.h = self.flat_dim[0] * self.flat_dim[1] * self.flat_dim[2]

        print(self.h)

        latent  = 10
        
        self.output = nn.Linear(self.h, latent)
        
        self.dist_layer = nn.Linear(latent, 1)
    
    def get_flat_dim(self):
        x = Variable(torch.randn(16, 1, 28, 28))

        x = F.relu(self.dropout[0](self.conv_layers[0](x)))
        x = F.max_pool2d(x,2)
        x = F.relu(self.dropout[1](self.conv_layers[1](x)))
        x = F.relu(self.dropout[2](self.conv_layers[2](x)))
        x = F.max_pool2d(x,2)
        x = F.relu(self.dropout[3](self.conv_layers[3](x)))

        x = F.avg_pool2d(x,2)

        return (x.size()[1:])
    
    def forward(self,x):

        x = F.relu(self.dropout[0](self.bn[0](self.conv_layers[0](x))))
        x = F.max_pool2d(x,2)
        x = F.relu(self.dropout[1](self.bn[1](self.conv_layers[1](x))))
        x = F.relu(self.dropout[2](self.bn[2](self.conv_layers[2](x))))
        x = F.max_pool2d(x,2)
        x = F.relu(self.dropout[3](self.bn[3](self.conv_layers[3](x))))

        x = F.avg_pool2d(x,2)
        
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        
        return self.output(x)
    
    def cost(self,x1,x2,y):

        mask = Variable(y.float())
        mask = mask.unsqueeze(1)
        
        z1 = model(x1)
        z2 = model(x2)

        P = F.sigmoid(self.dist_layer((z1-z2).abs()))
         
        L = mask* torch.log(P) + (1-mask) * torch.log(1-P)
    
        return((-L).mean())
    
 
    def predict_prob(self,x1,x2,y):

        mask = Variable(y.float())
                
        mask = mask
        
        z1 = model(x1)
        z2 = model(x2)
        
        P = F.sigmoid(self.dist_layer((z1-z2).abs()))
        P = P.squeeze()
        

        prob_same =  P[mask.byte()] 
        
        prob_diff  = (1-P)[(1-mask).byte()]

        return prob_same.data.numpy() , prob_diff.data.numpy()

load= True

if not load:

    print("Creating Model")
    model = SiameseNN()
    model = GRU_Model()


else:
    print("Loading Model")
    model = torch.load('model.p')



params = list(model.parameters())

for conv in model.conv_layers:
    params += conv.parameters()

opt = optim.Adam(params, lr=0.001)
    
print("Beginning Traiing")

def train():
    
    avg_loss = 0
    model.train()
    start = time.time()

    X1_train, X2_train, y_train , length = newDataset(np.expand_dims(train_loader.dataset.train_data.numpy() ,axis=1) / 255., train_loader.dataset.train_labels.numpy())

    for i in range(length):
            
        #data1, data2 ,target = train_loader.getBatch()
        data1, data2 ,target = getBatch(X1_train,X2_train,y_train,i)

        data1 = Variable(data1)
        data2 = Variable(data2)

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

    X1_test, X2_test, y_test , length = newDataset(np.expand_dims(test_loader.dataset.test_data.numpy() ,axis=1) / 255., test_loader.dataset.test_labels.numpy())


    for i in range(length):
         
        data1, data2 ,target = getBatch(X1_test,X2_test,y_test,i)
        #data1,data2,target =  test_loader.getBatch()

        data1 = Variable(data1)
        data2 = Variable(data2)

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

#for i in range(10):

#    train()
#    test()
    

#torch.save(model,open('model.p','wb'))
