import glob
import random
import os
import re
import sys
import matplotlib.pylab as plot
import numpy as np
import pathlib

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_TIR = []
        self.files_RGB = []

        try:
            file = open('Data_sorted.log', 'r')
            lines = file.readlines()
        except:
            print("Couldn't open file {}".format(root + 'utils/Data_sorted.log'))
            print("Consider running with flag '--sd'")
            sys.exit(255)

        for line in lines:
            if line.startswith('RGB_' + mode):
                match     = re.match('RGB_' + mode + ': (.*)', line)
                RGB_group = match.group(1)
                self.files_RGB = RGB_group.split(' ')
            elif line.startswith('TIR_' + mode):
                match     = re.match('TIR_' + mode + ': (.*)', line)
                TIR_group = match.group(1)
                self.files_TIR = RGB_group.split(' ')


        file.close()

        # self.files_TIR = sorted(glob.glob(os.path.join(root, '%s/TIR' % mode) + '/*.*'))
        # self.files_RGB = sorted(glob.glob(os.path.join(root, '%s/RGB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_TIR = plot.array(Image.open(self.files_TIR[index % len(self.files_TIR)]).convert("RGB"))
        item_RGB = plot.array(Image.open(self.files_RGB[index % len(self.files_RGB)]).convert("RGB"))
        item_RGB = item_RGB[113:1393, 33:993, :]

        return {'TIR': item_TIR, 'RGB': item_RGB}

    def __len__(self):
        return max(len(self.files_TIR), len(self.files_RGB))




def shuffle_data():
    rand_gen  = np.random.RandomState(1)
    RGB       = []
    TIR       = []
    root      = pathlib.Path(__file__).parent.parent.absolute()    # root of RGB2TIR
    data_dir  = os.path.join(root, "data")
    for shot_folder in os.listdir(data_dir):
        fus = os.path.join(data_dir, shot_folder, "fus")
        dc  = os.path.join(data_dir, shot_folder, "dc")
        RGB.append(glob.glob(dc + '/*.*'))
        TIR.append(glob.glob(dc + '/*.*'))

    TIR       = sorted(TIR)
    RGB       = sorted(RGB)
    n_samples = len(TIR)

    ## Generating a shuffled vector of indices
    indices = np.arange(n_samples)
    rand_gen.shuffle(indices)
    ## Split the indices into 80% train (full) / 20% test
    n_samples_train = int(n_samples * 0.8)
    train_indices   = indices[:n_samples_train]
    test_indices    = indices[n_samples_train:]
    ## %%%%%%%%%%%%%%% Your code here - End %%%%%%%%%%%%%%%%%

    ## Extract the sub datasets from the full dataset using the calculated indices
    TIR_train = [TIR[x] for x in train_indices]
    RGB_train = [RGB[x] for x in train_indices]

    TIR_test = [TIR[x] for x in test_indices]
    RGB_test = [RGB[x] for x in test_indices]

    data_file = open('Data_sorted.log', 'w')
    data_file.write("RGB_train: {}\n".format(' '.join(RGB_train)))
    data_file.write("TIR_train: {}\n".format(' '.join(RGB_train)))
    data_file.write("RGB_test: {}\n".format(' '.join(RGB_train)))
    data_file.write("TIR_test: {}\n".format(' '.join(RGB_train)))

    data_file.close()
    # files_RGB = sorted(glob.glob(os.path.join(root, "data") + '/*.*'))
