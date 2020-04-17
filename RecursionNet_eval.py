from __future__ import print_function, division

# Set variables for evaluation
num_workers = 30
batch_size = 32
load_path = f'./model/resnet50_200'

print("model: resnet50")
print('num_workers:', num_workers)
print('batch_size:', batch_size)
print('load_path:', load_path)

import multiprocessing
print('CPU cores:', multiprocessing.cpu_count(), '\n')

from torchvision import transforms, utils, datasets

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

from RecursionDS_eval import RecursionDataset

assert torch.cuda.is_available() # GPU must be available

eval_dataset = RecursionDataset(csv_file1='../recursion_data/test-labels/test.csv',
                                root_dir='../recursion_data/test-data',
                                #csv_file2='../recursion_data/test-labels/test_controls.csv',
                                phase = None)

model = models.resnet50(pretrained=False)

# Replace first and last layer
model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=2048, out_features=1139, bias=True)

# Load parameters
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

# put model on GPU
model.cuda()

since = time.time()

model.eval()
dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

loop = tqdm(total=len(dataloader), position=0, file=sys.stdout)
predictions = pd.DataFrame({'id_code': [], 'sirna': []})

for batch, (id_code, x1, x2) in enumerate(dataloader):
  x1, x2 = x1.cuda(non_blocking=True), x2.cuda(non_blocking=True)

  with torch.no_grad():
    y_hat1 = model(x1)
    y_hat2 = model(x2)
    
    y_hat1 = y_hat1/torch.sum(y_hat1)
    y_hat2 = y_hat2/torch.sum(y_hat2)
    
    y_hat = y_hat1 + y_hat2
    _, predicted = torch.max(y_hat, 1)
  
  new_pred = pd.DataFrame({'id_code':id_code, 'sirna':predicted.cpu()})
  predictions = predictions.append(new_pred, ignore_index=True, sort=False)
  loop.update(1)
     
loop.close()

time_elapsed = time.time() - since
print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

predictions.to_csv(f'predictions_resnet50_53acc.csv', index=False)
