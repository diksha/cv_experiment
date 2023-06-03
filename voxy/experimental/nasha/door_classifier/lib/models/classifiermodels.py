##
## Copyright 2020-2021 Voxel Labs, Inc.
## All rights reserved.
##
## This document may not be reproduced, republished, distributed, transmitted,
## displayed, broadcast or otherwise exploited in any manner without the express
## prior written permission of Voxel Labs, Inc. The receipt or possession of this
## document does not convey any rights to reproduce, disclose, or distribute its
## contents, or to manufacture, use, or sell anything that it may describe, in
## whole or in part.
##
import torchvision.models as models
from torch import nn
import torch
import copy

class SelfAttention(nn.Module):
    
    def __init__(self,input_dim: int):
        super(SelfAttention,self).__init__()
        self.to_query = nn.Conv2d(in_channels = input_dim , out_channels = input_dim//8 , kernel_size= 1)
        self.to_key = nn.Conv2d(in_channels = input_dim , out_channels = input_dim//8 , kernel_size= 1)
        self.to_value = nn.Conv2d(in_channels = input_dim , out_channels = input_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #
    
    def forward(self,x):
        bs, c, w, h = x.size()
        Q  = self.to_query(x).view(bs, -1, w * h).permute(0,2,1) 
        K =  self.to_key(x).view(bs,-1,w * h)
        E =  torch.bmm(Q,K) 
        A = self.softmax(E) 
        V = self.to_value(x).view(bs, -1, w * h) 
        Y = torch.bmm(V, A.permute(0,2,1))
        Y = Y.view(bs, c, w, h)
        out = self.gamma * Y + x
        return out, A
    
class AttentionResnet50(nn.Module):
    
    def __init__(self, num_classes: int):
        super(AttentionResnet50,self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.att_1 = SelfAttention(512)
        self.att_2 = SelfAttention(1024)
        self.att_3 = SelfAttention(2048)
        self.dropout=nn.Dropout(0.2)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self,x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x, _ = self.att_1(x)
        x = self.dropout(x)
        x = self.resnet.layer3(x)
        x, _ = self.att_2(x)
        x = self.dropout(x)
        x = self.resnet.layer4(x)
        x, _ = self.att_3(x)
        x = self.dropout(x)
        x = self.resnet.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.resnet.fc(x)
        return x

class VanillaResnet50(nn.Module):
    def __init__(self, num_classes: int, freeze_layers : bool):
        super(VanillaResnet50,self).__init__()
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = not(freeze_layers)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)