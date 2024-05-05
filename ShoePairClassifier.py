import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self): 
        super(CNN, self).__init__()
        n = 8
        kernel_size = 5
        padding = 2

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(in_channels=n,out_channels=2*n,kernel_size=kernel_size,padding=padding)
        self.bn2 = nn.BatchNorm2d(2*n)
        self.conv3 = nn.Conv2d(in_channels=2*n,out_channels=4*n,kernel_size=kernel_size,padding=padding)
        self.bn3 = nn.BatchNorm2d(4*n)
        self.conv4 = nn.Conv2d(in_channels=4*n,out_channels=8*n,kernel_size=kernel_size,padding=padding)
        self.bn4 = nn.BatchNorm2d(8*n)
        self.fc1 = nn.Linear(8*n*28*14,100)
        self.bn5 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,2)
    
    def forward(self,inp):
        inp = F.relu(self.bn1(self.conv1(inp)))
        inp = F.max_pool2d(inp,kernel_size= 2)

        inp = F.relu(self.bn2(self.conv2(inp)))
        inp = F.max_pool2d(inp,kernel_size= 2)

        inp = F.relu(self.bn3(self.conv3(inp)))
        inp = F.max_pool2d(inp,kernel_size= 2)

        inp = F.relu(self.bn4(self.conv4(inp)))
        inp = F.max_pool2d(inp,kernel_size= 2)

        inp = inp.reshape(-1,8*8*28*14)
        inp = F.relu(self.bn5(self.fc1(inp)))
        inp = self.fc2(inp)
        return inp

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        n = 4      
        kernel_size = 5
        padding = 2

        self.conv1 = nn.Conv2d(in_channels=6,out_channels=n,kernel_size=kernel_size,padding=padding)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(in_channels=n,out_channels=2*n,kernel_size=kernel_size,padding=padding)
        self.bn2 = nn.BatchNorm2d(2*n)
        self.conv3 = nn.Conv2d(in_channels=2*n,out_channels=4*n,kernel_size=kernel_size,padding=padding)
        self.bn3 = nn.BatchNorm2d(4*n)
        self.conv4 = nn.Conv2d(in_channels=4*n,out_channels=8*n,kernel_size=kernel_size,padding=padding)
        self.bn4 = nn.BatchNorm2d(8*n)
        self.fc1 = nn.Linear(8*n*(14**2),100)
        self.bn5 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100,2)

    # TODO: complete this class
    def forward(self,inp):# Do NOT change the signature of this function
        inp = torch.cat((inp[:,:,:224,:],inp[:,:,224:,:]),1)
        inp = F.relu(self.bn1(self.conv1(inp)))
        inp = F.max_pool2d(inp,kernel_size=2)

        inp = F.relu(self.bn2(self.conv2(inp)))
        inp = F.max_pool2d(inp,kernel_size=2)

        inp = F.relu(self.bn3(self.conv3(inp)))
        inp = F.max_pool2d(inp,kernel_size=2)

        inp = F.relu(self.bn4(self.conv4(inp)))
        inp = F.max_pool2d(inp,kernel_size=2)

        inp = inp.reshape(-1,8*4*(14**2))
        inp = F.relu(self.bn5(self.fc1(inp)))
        inp = self.fc2(inp)
        
        return inp