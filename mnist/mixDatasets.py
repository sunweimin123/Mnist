#coding:utf-8
from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import os
import codecs
from PIL import Image
import torchvision.transforms as transforms
import struct
class MyDataset(data.Dataset):
    def __init__(self, train=True):
        self.root = 'mnist_dataset'

        if train:
            self.labels_path = os.path.join(self.root, 'train-labels.idx1-ubyte')
            self.images_path = os.path.join(self.root, 'train-images.idx3-ubyte')
            self.num = 60000
        else:
            self.labels_path = os.path.join(self.root, 't10k-labels.idx1-ubyte')
            self.images_path = os.path.join(self.root, 't10k-images.idx3-ubyte')
            self.num = 10000
        self.images=read_image_file(self.images_path)
        self.labels=read_label_file(self.labels_path)






    def __getitem__(self, index):  # 返回的是tensor
        img = self.images[index]
        target = self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        transform1 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


        img = transform1(img)
        return img, target

    def __len__(self):
        return len(self.images)




def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)



def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])

        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


train_loader = torch.utils.data.DataLoader(MyDataset(train=True),
                                           batch_size=32, shuffle=True)
