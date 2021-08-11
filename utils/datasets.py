import glob
import random
import os
import re
import sys
# import matplotlib.pylab as plot
import numpy as np
import pathlib

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

class ImageDataset(Dataset):
    def __init__(self, TIR_transforms_=None, RGB_transforms_=None, unaligned=False, mode='train'):
        self.TIR_transform = transforms.Compose(TIR_transforms_)
        self.RGB_transform = transforms.Compose(RGB_transforms_)
        self.unaligned = unaligned
        self.files_TIR = []
        self.files_RGB = []
        self.mode = mode
        root = pathlib.Path(__file__).parent.parent.absolute()  # root of RGB2TIR

        try:
            file = open(os.path.join(root, 'bin/Data_sorted.txt'), 'r')
            lines = file.readlines()
        except:
            print("Couldn't open file {}".format(os.path.join(root, 'bin/Data_sorted.txt')))
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
                self.files_TIR = TIR_group.split(' ')

        file.close()

        # self.files_TIR = sorted(glob.glob(os.path.join(root, '%s/TIR' % mode) + '/*.*'))
        # self.files_RGB = sorted(glob.glob(os.path.join(root, '%s/RGB' % mode) + '/*.*'))

    def __getitem__(self, index):
        root = pathlib.Path(__file__).parent.parent.absolute()
        # item_TIR = self.transform(Image.open(self.files_TIR[index % len(self.files_TIR)]).convert('RGB'))
        TIR_file = self.files_TIR[index % len(self.files_TIR)]
        RGB_file = self.files_RGB[index % len(self.files_RGB)]

        item_TIR = self.TIR_transform(Image.open(TIR_file).convert('L'))

        Im_RGB = Image.open(RGB_file).convert('RGB').crop((33, 113, 993, 1393))
        Im_RGB = Im_RGB.resize((544, 720))
        item_RGB = self.RGB_transform(Im_RGB)



        output = os.path.basename(os.path.dirname(os.path.dirname(RGB_file))) + '.' + os.path.basename(RGB_file)
        return {'TIR': item_TIR, 'RGB': item_RGB, 'output': str(output)}

        # if self.mode == 'test':
        #     output = os.path.basename(os.path.dirname(os.path.dirname(RGB_file))) + '.' +  os.path.basename(RGB_file)
        #     return {'TIR': item_TIR, 'RGB': item_RGB, 'output': str(output)}
        #
        # else:
        #     return {'TIR': item_TIR, 'RGB': item_RGB}

    def __len__(self):
        return max(len(self.files_TIR), len(self.files_RGB))




def shuffle_data():
    rand_gen  = np.random.RandomState(23)
    RGB       = []
    TIR       = []
    root      = pathlib.Path(__file__).parent.parent.absolute()    # root of RGB2TIR
    data_dir  = os.path.join(root, "data/video")
    ## Get all TIR and RGB images
    for shot_folder in os.listdir(data_dir):
        fus = os.path.join(data_dir, shot_folder, "fus")
        dc  = os.path.join(data_dir, shot_folder, "dc")
        RGB.extend(glob.glob(dc + '/*'))
        TIR.extend(glob.glob(fus + '/*'))
    TIR = sorted(TIR)
    RGB = sorted(RGB)

    TIR = [TIR[x] for x in range(len(TIR)) if x%4 == 0]
    RGB = [RGB[x] for x in range(len(RGB)) if x%4 == 0]

    n_samples = len(TIR)

    ## Generating a shuffled vector of indices
    indices = np.arange(n_samples)
    rand_gen.shuffle(indices)
    ## Split the indices into 80% train (full) / 20% test
    n_samples_train = int(n_samples * 0.8)
    train_indices   = indices[:n_samples_train]
    test_indices    = indices[n_samples_train:]


    ## Extract the sub datasets from the full dataset using the calculated indices
    TIR_train = [TIR[x] for x in train_indices]
    RGB_train = [RGB[x] for x in train_indices]

    TIR_test = [TIR[x] for x in test_indices]
    RGB_test = [RGB[x] for x in test_indices]

    file_path = os.path.join(root, 'bin/Data_sorted.txt')
    data_file = open(file_path, 'w')
    data_file.write("RGB_train: {}\n".format(' '.join(RGB_train)))
    data_file.write("TIR_train: {}\n".format(' '.join(TIR_train)))
    data_file.write("RGB_test: {}\n".format(' '.join(RGB_test)))
    data_file.write("TIR_test: {}\n".format(' '.join(TIR_test)))

    data_file.close()


def get_list_of_files(mode='train'):
    files_TIR  = []
    files_RGB  = []
    final_list = []
    root=pathlib.Path(__file__).parent.parent.absolute()
    file_path = os.path.join(root, 'bin/Data_sorted.txt')

    try:
        file = open(file_path, 'r')
        lines = file.readlines()

    except:
        print("Couldn't open file {}".format(root + '/bin/Data_sorted.txt'))
        print("Consider running with flag '--sd'")
        sys.exit(255)

    for line in lines:
        if line.startswith('RGB_' + mode):
            match     = re.match('RGB_' + mode + ': (.*)', line) #
            RGB_group = match.group(1)
            files_RGB = RGB_group.split(' ')
        elif line.startswith('TIR_' + mode):
            match     = re.match('TIR_' + mode + ': (.*)', line)
            TIR_group = match.group(1)
            files_TIR = TIR_group.split(' ')

    file.close()

    for i in range(len(files_TIR)):
        if (i % 4 != 1):
            # continue
            pass
        db = {'TIR': files_TIR[i], 'RGB': files_RGB[i]}
        final_list.append(db)


    return final_list