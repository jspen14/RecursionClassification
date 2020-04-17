from __future__ import print_function, division

# Set variables for testing
num_workers = 30
batch_size = 32
n_epochs = 1
old_n_epochs = 0
lr = 0.2
save_path = f'./model/resnet50_{n_epochs+old_n_epochs}'
if old_n_epochs == 0:
  load_path = ''
else:
  load_path = f'./model/resnet50_{old_n_epochs}'
freeze = False
prop_train = 0.8
transformations = True

print('SGD w/ Restart')
print("model: resnet50")
print('num_workers:', num_workers)
print('batch_size:', batch_size)
print('learning rate:', lr)
print('n_epochs:', n_epochs)
print('load_path:', load_path)
print('save_path:', save_path)
print('freeze layers:', freeze)
print('Proportion used for training:', prop_train)

import multiprocessing
print('CPU cores:', multiprocessing.cpu_count(), '\n')

from torchvision import transforms, utils, datasets
if transformations == True:
  #rotations = [transforms.RandomRotation((90,90)),transforms.RandomRotation((180,180)),transforms.RandomRotation((270,270)), transforms.RandomRotation((0,0))]
  transformList = [transforms.Compose([transforms.ColorJitter((0,10), 5), transforms.RandomHorizontalFlip(), transforms.RandomRotation(180, fill = (0,))])]
  '''transformList = [transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation((90,90), fill = (0,))]),
                   transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation((180,180), fill = (0,))]),
                   transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation((270,270), fill = (0,))]),
                   transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation((0,0), fill = (0,))])]'''
else:
  transformList = None
print('Transformations:', transformations)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torchvision.models as models
from tqdm import tqdm
import pdb
import sys
import gc
from collections import OrderedDict
from skimage import io, transform
from PIL import Image
from torch.nn.parameter import Parameter
import pandas as pd
import random
import copy
import time

from IPython.core.ultratb import AutoFormattedTB

from RecursionDS_train import RecursionDataset

# from torchvision import transforms, utils, datasets

assert torch.cuda.is_available() # GPU must be available

train_dataset = RecursionDataset(csv_file1='../recursion_data/train-labels/train.csv',
                                root_dir='../recursion_data/train-data',
                                csv_file2='../recursion_data/train-labels/train_controls.csv',
                                phase = 'train', prop_train=prop_train, transform=transformList)
val_dataset = RecursionDataset(csv_file1='../recursion_data/train-labels/train.csv',
                                root_dir='../recursion_data/train-data',
                                csv_file2='../recursion_data/train-labels/train_controls.csv',
                                phase = 'val', prop_train=prop_train)

model = models.resnet50(pretrained=False)
model.load_state_dict(torch.load('models_original/resnet50_original'))

# Freeze all layers if true
if freeze:
  for param in model.parameters():
    param.requires_grad = False
# Replace first and last layer

model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=2048, out_features=train_dataset.n_classes, bias=True)
#model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#model.fc = nn.Linear(in_features=512, out_features=train_dataset.n_classes, bias=True)

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
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
#optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1)
accumulation_steps = 512 // batch_size

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
        pass #optimizer.zero_grad()

      x, y_truth = x.cuda(non_blocking=True), y_truth.cuda(non_blocking=True)
      y_truth = y_truth.type(torch.cuda.LongTensor).squeeze(1) #NOTE: making y_hat a 1D tensor for crossEnropyLoss function

      if phase == "train":
        y_hat = model(x)
        loss = objective(y_hat, y_truth) / accumulation_steps
        _, predicted = torch.max(y_hat, 1)
        loss.backward()

        if (batch) % accumulation_steps == 0:
          iters = len(dataloader)
          scheduler.step(epoch + batch / iters)
          optimizer.step()
          optimizer.zero_grad()
        elif batch+1 == len(dataloader):
          optimizer.zero_grad()

      else:
        with torch.no_grad():
          y_hat = model(x)
          loss = objective(y_hat, y_truth)
          _, predicted = torch.max(y_hat, 1)
      
      if phase == 'train':
        running_loss += loss.item() * x.size(0) * accumulation_steps
      else:
        running_loss += loss.item() * x.size(0)
      running_corrects += torch.sum(predicted == y_truth.data)
      running_total = (batch+1) * batch_size

      phase_loss = running_loss / running_total
      phase_acc = running_corrects.double() / running_total 

      loop.set_description('epoch: {}/{}, {} Loss: {:.4f}, {} Accuracy: {:.4f}'.format(epoch + 1, n_epochs, phase, phase_loss, phase, phase_acc)) 
	  
      loop.update(1)
     
    # Save loss and accuracy for reporting
    if phase == 'train':
        train_losses.append(phase_loss)
        train_acc.append(phase_acc.item())
    else:
        if phase_acc > best_acc: #Deep copy best model
          best_acc = phase_acc
          best_model_wts = copy.deepcopy(model.state_dict())          
        val_losses.append(phase_loss)
        val_acc.append(phase_acc.item())
    #log = 'epoch: {}/{}, {} Loss: {:.4f}, {} Accuracy: {:.4f}'.format(epoch + 1, n_epochs, phase, phase_loss, phase, phase_acc)
    #print(log)
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
report.to_csv(f'report_{old_n_epochs+n_epochs}.csv')
  
# Model Saving
state = {
    "epoch": n_epochs,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict()
}
torch.save(state, save_path)

gc.collect()
print('GPU Mem Used:', torch.cuda.memory_allocated(0) / 1e9)
