from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
# from torchvision import transforms, utils, datasets
import re
import random

class RecursionDataset(Dataset):
    """Recursion Dataset for Big Data Capstone."""

    def __init__(self, csv_file1, root_dir, csv_file2=None, transform=None, shuffle=True, phase = None, prop_train=0.8):
        """
        Args:
            csv_file1 (string): Path to the csv file with most annotations.
            root_dir (string): Directory with all the batch folders containing images.
            csv_file2 (string): Path to the csv file with control annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            shuffle (boolean): Optional shuffling, defaults to True
        """
        self.csv = pd.read_csv(csv_file1)
        if csv_file2 != None:
            csv2 = pd.read_csv(csv_file2).loc[:,'id_code':'sirna']
            self.csv = self.csv.append(csv2, ignore_index=True, sort=False)\
                         .reset_index(drop=True)
        self.csv['plate'] = 'Plate'+self.csv['plate'].astype(str) # Mimic folder naming for loading pics later
        
		# Remove pic observations (those without actual pic files) from csv
        missingPics = [['HUVEC-06_1_B18', 's2'], ["RPE-04_3_E04", 's1']] 
        for pic in missingPics:
            self.csv = self.csv[~((self.csv['id_code'] == pic[0]))]
        
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        # Generate full filename of image from csv file row info
        pathParts = self.csv.iloc[idx,:]
        pathGen = os.path.join(self.root_dir, pathParts['experiment'], pathParts['plate'])

        id = self.csv.iloc[idx,:].loc['id_code']
        pics = []
        first_missing = False
        for site in ['s1', 's2']:
          filenameGen = pathParts['well']+'_'+site+'_w'
  
          for i in range(1,7):
              filenameFull = filenameGen+str(i)+'.png'
              pathFull = os.path.join(pathGen, filenameFull)

              pic_gone = not os.path.isfile(pathFull)
              if pic_gone:
                  if site == 's1':
                    first_missing = True
                  break

              image = Image.open(pathFull)
              image = np.array(image)
  
              if i == 1:
                  totalTensor = torch.from_numpy(image).unsqueeze(0)
              else:
                  imageTensor = torch.from_numpy(image).unsqueeze(0)
                  totalTensor = torch.cat( (totalTensor, imageTensor), 0)
          if not pic_gone: # if current image exists
            pics.append(totalTensor.float())
            if first_missing: pics.append(pics[0])
          else: # current image is missing...
            id = 'EXTRA' #Mark to remove
            if first_missing: continue
            else: pics.append(pics[0])

        return id, pics[0], pics[1]

