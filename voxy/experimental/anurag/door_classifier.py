#
# Copyright 2020-2021 Voxel Labs, Inc.
# All rights reserved.
#
# This document may not be reproduced, republished, distributed, transmitted,
# displayed, broadcast or otherwise exploited in any manner without the express
# prior written permission of Voxel Labs, Inc. The receipt or possession of this
# document does not convey any rights to reproduce, disclose, or distribute its
# contents, or to manufacture, use, or sell anything that it may describe, in
# whole or in part.
#
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import models
from torchvision import transforms

train_data_dir = "/home/anurag_voxelsafety_com/data/door_state_classifier/train/"
val_data_dir = "/home/anurag_voxelsafety_com/data/door_state_classifier/test/"


train_transforms = transforms.Compose([transforms.ToTensor()])
val_transforms = transforms.Compose([transforms.ToTensor()])
train_data = datasets.ImageFolder(train_data_dir, transform=train_transforms)
val_data = datasets.ImageFolder(val_data_dir, transform=val_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=1)
valloader = torch.utils.data.DataLoader(val_data, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)

# for param in model.parameters():
#     param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 10),
    nn.LogSoftmax(dim=1),
)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.1)
model.to(device)


epochs = 3
steps = 0
running_loss = 0
print_every = 10
train_losses, val_losses = [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            val_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    val_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss / len(trainloader))
            val_losses.append(val_loss / len(valloader))
            print(
                f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"val loss: {val_loss/len(valloader):.3f}.. "
                f"val accuracy: {accuracy/len(valloader):.3f}"
            )
            running_loss = 0
            model.train()
