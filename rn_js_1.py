from __future__ import print_function, division

# Set variables for testing
num_workers = 30
batch_size = 32
n_epochs = 1
old_n_epochs = 0
lr = 3e-4
save_path = f'./model/resnet18_{n_epochs+old_n_epochs}'
if old_n_epochs == 0:
  load_path = ''
else:
  load_path = f'./model/resnet18_{old_n_epochs}'
freeze = True
prop_train=0.8

print("model: resnet18")
print('num_workers:', num_workers)
print('batch_size:', batch_size)
print('n_epochs:', n_epochs)
print('load_path:', load_path)
print('save_path:', save_path)
print('freeze layers:', freeze)
print('Proportion used for training:', prop_train)

import multiprocessing
print('CPUs:', multiprocessing.cpu_count(), '\n')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms, utils, datasets
import torchvision.models as models
from tqdm import tqdm
import pdb
import sys
import gc
from collections import OrderedDict
from skimage import io, transform
from torch.nn.parameter import Parameter
import pandas as pd
import random
import copy
import time

from IPython.core.ultratb import AutoFormattedTB

from RecursionDS import RecursionDataset

# from torchvision import transforms, utils, datasets

assert torch.cuda.is_available() # GPU must be available

train_dataset = RecursionDataset(csv_file1='../../recursion_data/train-labels/train.csv', root_dir='../../recursion_data/train-data', csv_file2='../../recursion_data/train-labels/train_controls.csv', phase = 'train', prop_train=prop_train)
val_dataset = RecursionDataset(csv_file1='../../recursion_data/train-labels/train.csv', root_dir='../../recursion_data/train-data', csv_file2='../../recursion_data/train-labels/train_controls.csv', phase = 'val', prop_train=prop_train)

model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('../BaseModels/resnet18-5c106cde.pth'))

# Freeze all layers if true
if freeze:
  for param in model.parameters():
    param.requires_grad = False
# Replace first and last layer
model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=train_dataset.n_classes, bias=True)

#collect which parameters to update to send to the optimizer (if not freezing, send all the params)
params_to_update = model.parameters()
print("Params to learn:")
if freeze == True:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print('\t', name)
else:
  params_to_update = model.parameters()
  print('\t', 'update all params')

# Load new parameters if listed
if load_path != '':
  state_loaded = torch.load(load_path)
  new_state_dict = OrderedDict()
  for k, v in state_loaded['state_dict'].items():
    name = k[7:] # remove module.
    new_state_dict[name] = v
  model.load_state_dict(new_state_dict)

# Use data parallelism if possible (use multiple GPUs)
if torch.cuda.device_count() > 1:
  print('Using', torch.cuda.device_count(), 'GPUs')
  model = nn.DataParallel(model)

# put model on GPU and prep objective, optimizer
model.cuda()
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(params_to_update, lr=lr)
if load_path != '':
  optimizer.load_state_dict(state_loaded['optimizer'])
since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

train_losses = []
train_acc = []
val_losses = []
val_acc = []

for epoch in range(n_epochs):
  
  # Each epoch has a training and validation phase
  for phase in ['train', 'val']:
    if phase == 'train':
          model.train()  # Set model to training mode
          dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    else:
          model.eval()   # Set model to evaluate mode
          dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)


    running_loss = 0.0
    running_corrects = 0

    loop = tqdm(total=len(dataloader), position=0, file=sys.stdout)

    for batch, (x, y_truth) in enumerate(dataloader):

      if phase == "train":
        optimizer.zero_grad()

      x, y_truth = x.cuda(non_blocking=True), y_truth.cuda(non_blocking=True)
      y_truth = y_truth.type(torch.cuda.LongTensor).squeeze(1) #NOTE: making y_hat a 1D tensor for crossEnropyLoss function

      if phase == "train":
        y_hat = model(x)
        loss = objective(y_hat, y_truth)
        _, predicted = torch.max(y_hat, 1)
        loss.backward()
        optimizer.step()

      else:
        with torch.no_grad():
          y_hat = model(x)
          loss = objective(y_hat, y_truth)
          _, predicted = torch.max(y_hat, 1)

      running_loss += loss.item() * x.size(0)
      running_corrects += torch.sum(predicted == y_truth.data)

      phase_loss = running_loss / len(dataloader.dataset)
      phase_acc = running_corrects.double() / len(dataloader.dataset) 

      loop.set_description('epoch: {}/{}, {} Loss: {:.4f}, {} Accuracy: {:.4f}'.format(epoch + 1, n_epochs, phase, phase_loss, phase, phase_acc)) 
	  
              # deep copy the model
      if phase == 'val' and phase_acc > best_acc:
          best_acc = phase_acc
          best_model_wts = copy.deepcopy(model.state_dict())
          
      loop.update(1)
    
    # Save loss and accuracy for reporting
    if phase == 'train':
        train_losses.append(phase_loss)
        train_acc.append(phase_acc.item())
    else:
        val_losses.append(phase_loss)
        val_acc.append(phase_acc.item())
    
    loop.close()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
model.load_state_dict(best_model_wts)

report = pd.DataFrame({
  'train_loss': train_losses,
  'train_acc': train_acc,
  'val_loss': val_losses,
  'val_acc': val_acc
})
report.to_csv('report.csv')
  
# Model Saving
state = {
    "epoch": n_epochs,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict()
}
torch.save(state, save_path)

gc.collect()
print('GPU Mem Used:', torch.cuda.memory_allocated(0) / 1e9)
