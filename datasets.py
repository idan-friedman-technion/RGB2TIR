import glob
import random
import os
import matplotlib.pylab as plot
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_TIR = sorted(glob.glob(os.path.join(root, '%s/TIR' % mode) + '/*.*'))
        self.files_RGB = sorted(glob.glob(os.path.join(root, '%s/RGB' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_TIR = plot.array(Image.open(self.files_TIR[index % len(self.files_TIR)]).convert("RGB"))
        item_RGB = plot.array(Image.open(self.files_RGB[index % len(self.files_RGB)]).convert("RGB"))
        item_RGB = item_RGB[113:1393, 33:993, :]

        return {'TIR': item_TIR, 'RGB': item_RGB}

    def __len__(self):
        return max(len(self.files_TIR), len(self.files_RGB))