
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class GRU_Model(nn.Module):


    def __init__(self,h, glove, embed_dim , num_labels):
        super(GRU_Model, self).__init__()

        self.h = h
        self.embed = nn.Embedding(glove.size()[0], glove.size()[1], padding_idx=0 )
        self.embed.weight = nn.Parameter(glove )

        self.gru = nn.GRU(glove.size()[1], h, 1, batch_first=True , dropout= .5)

        self.output_layer = nn.Linear(h, embed_dim,bias=False)
        self.dist_layer = nn.Linear(embed_dim, 1,bias=False)

        self.params = list(self.embed.parameters()) + list(self.output_layer.parameters()) + list(self.gru.parameters())


    def forward(self,x):

        h0 = Variable(torch.zeros(1, x.size()[0], self.h))

        E = self.embed(x)
        
        z = self.gru(E, h0)[0][:, -1, :]

        y_hat = self.output_layer(z)

        return y_hat

    def cost(self,x1,x2,y):

        mask = y
        mask = mask.unsqueeze(1)
        
        z1 = self.forward(x1)
        z2 = self.forward(x2)

        P = F.sigmoid(self.dist_layer((z1-z2).abs()))
         
        L = mask* torch.log(P) + (1-mask) * torch.log(1-P)
    
        return((-L).mean())
    
 
    def predict_prob(self,x1,x2,y):

        mask = y
                        
        z1 = self.forward(x1)
        z2 = self.forward(x2)
        
        P = F.sigmoid(self.dist_layer((z1-z2).abs()))
        P = P.squeeze()
        
        prob_same =  P[mask.byte()] 
        
        prob_diff  = (1-P)[(1-mask).byte()]

        return prob_same.data.numpy() , prob_diff.data.numpy()
        

    def predict(self, z1, x, y):

        z2 = self.forward(x)
        z2 = z2.expand_as(z1)

        P = F.sigmoid(self.dist_layer((z1 - z2).abs()))

        argmax = torch.max(P,0)[1].data.numpy()[0,0]

        return y[argmax]




class CNN(nn.Module):

    def __init__(self,glove,num_out,seq_len):
        super(CNN, self).__init__()

        self.seq_len = seq_len

        self.embed = nn.Embedding(glove.size()[0], glove.size()[1], padding_idx=0)
        self.embed.weight = nn.Parameter(glove)

        self.conv1 = nn.Conv1d(in_channels=50, out_channels=100, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)
        self.conv5 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3)

        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)
        self.drop4 = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(100)
        self.bn5 = nn.BatchNorm1d(100)

        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        self.pool3 = nn.MaxPool1d(2)
        self.pool4 = nn.MaxPool1d(2)


        self.flat_dim = self.get_flat_dim()

        self.output_layer = nn.Linear(self.flat_dim, num_out,bias=False)

        self.params = self.parameters()

    def get_flat_dim(self):

        x = Variable(torch.ones(32,self.seq_len)).long()

        E = self.embed(x)

        E = E.transpose(1, 2).contiguous()

        h = F.relu(self.bn1(self.conv1(E)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool1(h)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.pool2(h)
        h = F.relu(self.bn5(self.conv5(h)))
        h = self.pool3(h)

        print(h.size()[1] , h.size()[2])

        return(h.size()[1] * h.size()[2])


    def forward(self,x):

        E = self.embed(x)

        E = E.transpose(1, 2).contiguous()

        h = F.relu(self.bn1(self.conv1(E)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool1(h)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.pool2(h)
        h = F.relu(self.bn5(self.conv5(h)))
        h = self.pool3(h)

        h = h.view(-1,self.flat_dim)

        return F.sigmoid(self.output_layer(h))
