import numpy as np
import torch

class BatchLoader():
    def __init__(self, X, y, batch_size=64):

        self.batch_size = batch_size

        self.num_labels = y.max() + 1

        self.X = X
        self.y = y

        self.priors = [(y == i).sum() / float(X.shape[0]) for i in range(self.num_labels)]
        self._X = [X[y == i] for i in range(self.num_labels)]

        self.batch_index = 0

        self.newDataset()


    def newDataset(self):

        # Get pairs of data with different labels
        shuffle = np.random.permutation(self.X.shape[0])

        m = (self.y != self.y[shuffle])

        x1 = self.X[m]
        x2 = self.X[shuffle][m]

        y_new = np.zeros(x1.shape[0])

        X1 = [x1]
        X2 = [x2]

        # add pairs of data with the same label
        for i in range(self.num_labels):
            num_sample = np.floor(self.priors[i] * len(m))
            X1.append(self._X[i][np.random.randint(0, self._X[i].shape[0], num_sample,dtype='int')])
            X2.append(self._X[i][np.random.randint(0, self._X[i].shape[0], num_sample,dtype='int')])

        X1 = np.vstack(X1)
        X2 = np.vstack(X2)

        y_new = np.concatenate([y_new, np.ones(X1.shape[0] - y_new.shape[0])], axis=0)

        shuffle = np.random.permutation(X1.shape[0])[0:int(X1.shape[0]//3) ]

        self.X1 = torch.from_numpy(X1[shuffle]).float()
        self.X2 = torch.from_numpy(X2[shuffle]).float()
        self._y = torch.from_numpy(y_new[shuffle])

        self.length = int(np.ceil(self.X1.size()[0] / self.batch_size))

        self.batch_index=0

    def reset(self):
        self.batch_index=0

    def getBatch(self):

        if (self.batch_index + 1) * self.batch_size > self.X1.size()[0]:
            low = self.batch_index * self.batch_size
            high = self.X1.size()[0]
        else:
            low = self.batch_index * self.batch_size
            high = (self.batch_index + 1) * self.batch_size

        self.batch_index += 1

        return (self.X1[low:high], self.X2[low:high] ,  self._y[low:high])


def newDataset(X, y):

    num_labels = y.max() + 1

    priors = [(y == i).sum() / float(X.shape[0]) for i in range(num_labels)]
    _X = [X[y == i] for i in range(num_labels)]

    shuffle = np.random.permutation(X.shape[0])
    m = (y != y[shuffle])

    x1 = X[m]
    x2 = X[shuffle][m]
    
    y_new = np.zeros(x1.shape[0])

    X1 = [x1]
    X2 = [x2]
    
    # add pairs of data with the same label
    
    for i in range(num_labels):
    
        num_sample = np.floor(priors[i] * len(m))
        
        X1.append( _X[i][np.random.randint(0, _X[i].shape[0], num_sample,dtype='int')])
        
        X2.append(_X[i][np.random.randint(0, _X[i].shape[0], num_sample,dtype='int')])
        
        
    X1 = np.vstack(X1)
        
    X2 = np.vstack(X2)

    y_new = np.concatenate([y_new, np.ones(X1.shape[0] - y_new.shape[0])], axis=0)
        
    shuffle = np.random.permutation(X1.shape[0])[0:int(X1.shape[0]//3) ]
        
    X1 = torch.from_numpy(X1[shuffle]).float()
        
    X2 = torch.from_numpy(X2[shuffle]).float()
    
    _y = torch.from_numpy(y_new[shuffle])

    length = int(np.ceil(X1.size()[0] / 64))


    return(X1,X2,_y,length)


def getBatch(X1,X2,y,batch_index):

    batch_size = 64
    if (batch_index + 1) * batch_size > X1.size()[0]:
    
        low = batch_index * batch_size
        
        high = X1.size()[0]
        
    else:
    
        low = batch_index * batch_size
        
        high = (batch_index + 1) * batch_size

    return (X1[low:high], X2[low:high] , y[low:high])

