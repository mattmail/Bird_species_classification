#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:49:52 2018

@author: matthis
"""

from torchvision import models
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F


class Net6(nn.Module):
    def __init__(self, n_classes = 20):
        super(Net6, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 256, kernel_size=3)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.drop1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(50176, 1024)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.drop1(F.max_pool2d(F.relu(self.conv2(F.relu(self.conv1(x)))),2))
        x = self.drop1(F.max_pool2d(F.relu(self.conv4(F.relu(self.conv3(x)))),2))
        x = self.drop1(F.max_pool2d(F.relu(self.conv6(F.relu(self.conv5(x)))),2))
        x = x.view(-1, 50176)
        x = self.drop2(F.relu(self.fc1(x)))
        return self.fc2(x)

class Resnet(nn.Module):
    
    def __init__(self, num_classes =20):
        super(Resnet, self).__init__()
        self.model = models.resnet152(pretrained=True)

        for params in list(self.model.parameters())[:-50]:
            params.requires_grad = False
            
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1024)
        self.linear = nn.Linear(1024, num_classes)
    
    
    def forward(self, x):
        x = self.linear(F.relu(self.model(x)))
        return (x)
    
class InceptionV2(nn.Module):
    
    def __init__(self, num_classes =20):
        super(InceptionV2, self).__init__()
        self.model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
        for params in self.model.parameters():
            params.requires_grad = False
            
        self.linear1 = nn.Linear(1000, num_classes)
    
    
    def forward(self, x):
        x = self.model.forward(x)
        x = F.relu(self.linear1(x))
        return (x)

class Pnasnet5(nn.Module):
    
    def __init__(self, num_classes =20):
        super(Pnasnet5, self).__init__()
        self.model = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained='imagenet')
        for params in self.model.parameters():
            params.requires_grad = False
            
        self.linear1 = nn.Linear(1000, num_classes)
        self.drop = nn.Dropout(p=0.2)
    
    
    def forward(self, x):
        x = self.model.forward(x)
        x = F.relu(self.drop(self.linear1(x)))
        return (x)

class Densenet(nn.Module):
    
    def __init__(self, num_classes=20):
        super(Densenet, self).__init__()
        self.model = models.densenet201(pretrained=True)
        
        for params in list(self.model.features.parameters())[:-100]:
            params.requires_grad = False
            
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return (self.model(x))
    