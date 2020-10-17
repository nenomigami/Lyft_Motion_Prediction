#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:37:50 2020

@author: hoyun
"""

#1 try mini-batch level checkpoint saver

import itertools
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torchvision import datasets,transforms
from torch.utils.data import Dataset, DataLoader

from src.utils import FixedShuffleDataset

import numpy as np
import random
import os
import pickle as pkl
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

class TestFixedShuffleDataset(object):
    
    def test_dataloader_fixed(self):
        
        trn_dataset = datasets.MNIST('../mnist_data/',
                                     download=True,
                                     train=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ])) 
        batch_size = 32
        
        transformed_dataset1 = FixedShuffleDataset(trn_dataset,0)
        transformed_dataset2 = FixedShuffleDataset(trn_dataset,49 * 32)   
        
        transformed_loader1 = DataLoader(transformed_dataset1,
                             batch_size = batch_size,
                             shuffle = False)
        transformed_loader2 = DataLoader(transformed_dataset2,
                             batch_size = batch_size,
                             shuffle = False)
        
        tr_it1 = iter(transformed_loader1)
        tr_it2 = iter(transformed_loader2)

        for i in range(50):
            x1, y1 = next(tr_it1)        
        x2, y2 = next(tr_it2)
        assert torch.equal(x1,x2)  

    def test_on_saving_batch_level_first_part(self):
        
        device = torch.device("cuda:0")
        class CNNClassifier(nn.Module):
            
            def __init__(self):
                super(CNNClassifier, self).__init__()
                conv1 = nn.Conv2d(1,6,5,1)
                
                pool1 = nn.MaxPool2d(2)
                conv2 = nn.Conv2d(6, 16, 5,1)
                
                pool2 = nn.MaxPool2d(2)
                
                self.conv_module = nn.Sequential(
                    conv1,
                    nn.ReLU(),
                    pool1,
                    conv2,
                    nn.ReLU(),
                    pool2)
                
                fc1 = nn.Linear(16*4*4, 120)
                fc2 = nn.Linear(120, 84)
                fc3 = nn.Linear(84, 10)
                
                self.fc_module = nn.Sequential(
                    fc1,
                    nn.ReLU(),
                    fc2,
                    nn.ReLU(),
                    fc3)
            
                self.conv_module = self.conv_module.to(device)
                self.fc_module = self.fc_module.cuda()
            
            def forward(self, x):
                out = self.conv_module(x)
                dim = 1
                for d in out.size()[1:]:
                    dim = dim * d
                out = out.view(-1, dim)
                out = self.fc_module(out)
                return F.softmax(out,dim = 1)

        trn_dataset = datasets.MNIST('../mnist_data/',
                                     download=True,
                                     train=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ])) 
        batch_size = 32
        transformed_dataset = FixedShuffleDataset(trn_dataset,0)       
        transformed_loader = DataLoader(transformed_dataset,
                                        batch_size = batch_size,
                                        shuffle = False)

        cnn = CNNClassifier()
        criterion = nn.CrossEntropyLoss()
        learning_rate = 1e-3
        optimizer = optim.Adam(cnn.parameters(), lr = learning_rate)
        num_step = 50
        
        progress_bar = tqdm(range(num_step))
        trn_loss_list = []
        tr_it = iter(transformed_loader)
        for i in progress_bar:
            data = next(tr_it)
            x, label = data
            x = x.to(device)
            label = label.to(device)
            cnn.train()
            
            model_output = cnn(x)
            loss = criterion(model_output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            trn_loss_list.append(loss.item())
        
        torch.save({'weight': cnn.state_dict(),
                    'optim' : optimizer.state_dict()},
                   "seed42rep50_checkpoint.pt")
        with open("seed42rep100.pkl", "rb") as f:
            #pkl.dump(trn_loss_list,f)
            rep100 = pkl.load(f)
        assert rep100[:50] == trn_loss_list
        
    def test_on_saving_batch_level_last_part(self):
        
        device = torch.device("cuda:0")
        class CNNClassifier(nn.Module):
            
            def __init__(self):
                super(CNNClassifier, self).__init__()
                conv1 = nn.Conv2d(1,6,5,1)
                
                pool1 = nn.MaxPool2d(2)
                conv2 = nn.Conv2d(6, 16, 5,1)
                
                pool2 = nn.MaxPool2d(2)
                
                self.conv_module = nn.Sequential(
                    conv1,
                    nn.ReLU(),
                    pool1,
                    conv2,
                    nn.ReLU(),
                    pool2)
                
                fc1 = nn.Linear(16*4*4, 120)
                fc2 = nn.Linear(120, 84)
                fc3 = nn.Linear(84, 10)
                
                self.fc_module = nn.Sequential(
                    fc1,
                    nn.ReLU(),
                    fc2,
                    nn.ReLU(),
                    fc3)
            
                self.conv_module = self.conv_module.to(device)
                self.fc_module = self.fc_module.cuda()
            
            def forward(self, x):
                out = self.conv_module(x)
                dim = 1
                for d in out.size()[1:]:
                    dim = dim * d
                out = out.view(-1, dim)
                out = self.fc_module(out)
                return F.softmax(out,dim = 1)

        trn_dataset = datasets.MNIST('../mnist_data/',
                                     download=True,
                                     train=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ])) 
        batch_size = 32
        transformed_dataset = FixedShuffleDataset(trn_dataset,50*batch_size)       
        transformed_loader = DataLoader(transformed_dataset,
                                        batch_size = batch_size,
                                        shuffle = False)

        cnn = CNNClassifier()
        checkpoint = torch.load("seed42rep50_checkpoint.pt")
        cnn.load_state_dict(checkpoint["weight"])
        cnn.train()
        
        
        criterion = nn.CrossEntropyLoss()
        learning_rate = 1e-3
        optimizer = optim.Adam(cnn.parameters(), lr = learning_rate)
        optimizer.load_state_dict(checkpoint["optim"])     
        optimizer.state_dict()
        num_step = 50
        
        progress_bar = tqdm(range(num_step))
        trn_loss_list = []
        tr_it = iter(transformed_loader)
        for i in progress_bar:
            data = next(tr_it)
            x, label = data
            x = x.to(device)
            label = label.to(device)
            cnn.train()
            
            model_output = cnn(x)
            loss = criterion(model_output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            trn_loss_list.append(loss.item())
        
        with open("seed42rep100.pkl", "rb") as f:
            rep100 = pkl.load(f)
        assert rep100[50:] == trn_loss_list
