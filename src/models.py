'''
Code from Double Bubble, Toil against Trouble: Enhancing Certified Robustness with Transitivity
By Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani and Benjamin I.P. Rubinstein
'''

import torch

from typing import *
from datasets import dataset_load

import torchvision.utils
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class NormalizeLayer(torch.nn.Module):
    def __init__(self, mean, std, cuda=True):
        """
        mean: the channel means
        std: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(mean)#.cuda()
        self.sds = torch.tensor(std)#.cuda()
        if cuda:
            self.means = self.means.cuda()
            self.sds = self.sds.cuda()

    def forward(self, input: torch.tensor):
        self.means, self.sds = self.means.to(input.device), self.sds.to(input.device)
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds   
                                     
                                     
def model_settings(dataset: str, args):                                
    if dataset == 'mnist':
        return _mnist(args)
    elif dataset == 'cifar10':
        return _cifar10(args)
    elif dataset == 'tinyimagenet':
        return _tinyimagenet(args)


def _mnist(args):
    if args.batch_size == 0:
        batch_size = 128
    else: 
        batch_size = args.batch_size
        
    if args.lr == 0:
        lr = 0.001
    else:
        lr = args.lr
        
    norm_layer = NormalizeLayer(mean=[0.1307], std=[0.3081], cuda=False)

    train_loader  = torch.utils.data.DataLoader(dataset=dataset_load('mnist', 'train'),
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=10)

    val_loader = torch.utils.data.DataLoader(dataset=dataset_load('mnist', 'test'),
                                             batch_size=batch_size, 
                                             shuffle=True, num_workers=10)

    test_loader = torch.utils.data.DataLoader(dataset=dataset_load('mnist', 'test'),
                                             batch_size=1,
                                             shuffle=True, num_workers=8)

    model = models.resnet18(num_classes=10)
    model = torch.nn.Sequential(norm_layer, model)
    
    cuda_device_count = torch.cuda.device_count()
    if args.parallel  == 'always':
        model = torch.nn.DataParallel(model, device_ids=[i in range(cuda_device_count)])    
        device = torch.device("cpu")
    else: 
        device = torch.device("cuda" if cuda_device_count > 0 else "cpu")        
        model = model.to(device)    

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, loss, optimizer, None, train_loader, val_loader, test_loader, device, 10
    
def _cifar10(args):
    if args.batch_size == 0:
        batch_size = 150
    else: 
        batch_size = args.batch_size
        
    if args.lr == 0:
        lr = 0.001
    else:
        lr = args.lr
        
    norm_layer = NormalizeLayer(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261], cuda=False)
    

    train_loader = torch.utils.data.DataLoader(dataset=dataset_load('cifar10', 'train'),
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_load('cifar10', 'test'),
                                              batch_size=1,
                                              shuffle=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_load('cifar10', 'test'),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers = 8)
                                   
    model = models.resnet18(num_classes=10)
    model = torch.nn.Sequential(norm_layer, model)
    
    cuda_device_count = torch.cuda.device_count()
    if args.parallel  == 'always':
        model = torch.nn.DataParallel(model, device_ids=[i in range(cuda_device_count)])    
        device = torch.device("cpu")
    else:   
        device = torch.device("cuda" if cuda_device_count > 0 else "cpu")        
        model = model.to(device)  
    
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)                                           
        
    return model, loss, optimizer, None, train_loader, val_loader, test_loader, device, 10
    
def _tinyimagenet(args):
    if args.batch_size == 0:
        batch_size = 128
    else: 
        batch_size = args.batch_size
        
    if args.lr == 0:
        lr = 0.1
    else:    
        lr = args.lr
        

    normalize_layer = NormalizeLayer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], cuda=False) 
    train_loader = torch.utils.data.DataLoader(dataset=dataset_load('tinyimagenet', 'train'),
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_load('tinyimagenet', 'test'),
                                              batch_size=1,
                                              shuffle=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_load('tinyimagenet', 'test'),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers = 8)
    
    model = models.resnet18(num_classes=200)
    model.conv1 = torch.nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    model.fc.out_features = 200
    model = torch.nn.Sequential(normalize_layer, model)
    
    cuda_device_count = torch.cuda.device_count()
    if args.parallel  == 'always':   
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(cuda_device_count)])    
        device = torch.device("cpu")
    else: 
        device = torch.device("cuda" if cuda_device_count > 0 else "cpu")        
        model = model.to(device) 

    
    loss = nn.CrossEntropyLoss()    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)#001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return model, loss, optimizer, lr_scheduler, train_loader, val_loader, test_loader, device, 200
            


