from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
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
            self.csv = pd.concat([self.csv, csv2])\
                         .reset_index(drop=True)
        self.csv['plate'] = 'Plate'+self.csv['plate'].astype(str) # Mimic folder naming for loading pics later

        # Create variable for both sites 1 and 2 of each well
        self.csv['site'] = 's1'
        csv_copy = self.csv.copy()
        csv_copy['site'] = 's2'
        self.csv = pd.concat([self.csv, csv_copy])\
                     .sort_values(['id_code', 'site'])\
                     .reset_index(drop=True)

                # Remove pic observations (those without actual pic files) from csv
        missingPics = [['HUVEC-06_1_B18', 's2'], ["RPE-04_3_E04", 's1']]
        for pic in missingPics:
            self.csv = self.csv[~((self.csv['id_code'] == pic[0]) & (self.csv['site'] == pic[1]))]

        self.root_dir = root_dir
        self.transform = transform

        # Calculate number of unique classes
        self.n_classes = len(np.unique(self.csv['sirna']))

        #additions for validation set
        exp_list = self.csv.experiment.unique()
        random.Random(4).shuffle(exp_list) #no bias in selection, set seed for replicating results
        exp_train = exp_list[:round(prop_train*len(exp_list))] #set to 100*prop_train% of the total train data we have
        if phase == 'train':
          self.csv = self.csv[self.csv.experiment.isin(exp_train)]
          self.csv.sort_values(['id_code', 'site'])\
                  .reset_index(drop = True)
        elif phase == 'val':
          self.csv = self.csv[-self.csv.experiment.isin(exp_train)]
          self.csv.sort_values(['id_code', 'site'])\
                  .reset_index(drop = True)
        else: #just use whole dataset
          self.csv = self.csv

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, idx):
        # Generate full filename of image from csv file row info
        pathParts = self.csv.iloc[idx,:]
        pathGen = os.path.join(self.root_dir, pathParts['experiment'], pathParts['plate'])
        filenameGen = pathParts['well']+'_'+pathParts['site']+'_w'

        for i in range(1,7):
            filenameFull = filenameGen+str(i)+'.png'
            pathFull = os.path.join(pathGen, filenameFull)

            if not os.path.isfile(pathFull):
                print("Path: ", pathFull)
                return torch.zeros(size=(0,1)), torch.zeros(size=(0,1))

            image = io.imread(pathFull)
            if i == 1:
                totalTensor = torch.from_numpy(image).unsqueeze(0)
            else:
                imageTensor = torch.from_numpy(image).unsqueeze(0)
                totalTensor = torch.cat( (totalTensor, imageTensor), 0)

        try:
            sirna = self.csv.iloc[idx,:].loc['sirna']
        except:
            sirna = 1139

        if sirna=='UNTREATED': sirna = 1138
        else: sirna = float(re.search('[0-9]+', sirna).group())
        sirnaTensor = torch.tensor([sirna])
        #print("tT: ", totalTensor.shape, " sT:", sirnaTensor.shape)

        # Apply transformation
        if self.transform != None:
            toPil = transforms.ToPILImage()
            randTransform = transforms.RandomOrder(self.transform)
            toTensor = transforms.ToTensor()
            tensorHalves = torch.split(totalTensor, 3, dim=0)
            for half in tensorHalves:
              half = toTensor(randTransform(toPil(half)))
            totalTensor = torch.cat(tensorHalves, dim=0)

        return totalTensor.float(), sirnaTensor.float()
