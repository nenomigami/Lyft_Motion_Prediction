#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 13:16:16 2020

@author: hoyun
"""
from src.common_import import *
import yaml

def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f)
    print("saved as yaml")
    return

def load_yaml(path):
    with open(path) as f:
        temp = yaml.load(f, Loader=yaml.FullLoader)
    print("loaded yaml")
    return temp

class FixedShuffleDataset(Dataset):
    
    def __init__(self, dataset, nth_batch):
        super().__init__()
        self.dataset = dataset
        self.nth_batch = nth_batch
        np.random.seed(42)
        self.random_shuffle = np.random.permutation(np.arange(len(self.dataset)))
        self.random_shuffle = np.concatenate([self.random_shuffle[self.nth_batch:],
                                              self.random_shuffle[:self.nth_batch]])
    def __getitem__(self, index):
        idx = self.random_shuffle[index]
        return self.dataset[idx]
        
    def __len__(self):
        return len(self.dataset)