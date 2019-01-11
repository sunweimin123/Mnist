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

        self.images=read_image_file(self.images_path,self.num)
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
    with open(path, 'rb')as lpath:
        magic, n = struct.unpack('>II', lpath.read(8))
        train_labels = np.fromfile(lpath, dtype=np.uint8).astype(np.int64)
        labels = torch.from_numpy(train_labels)
        return labels

def read_image_file(path,numb):
    transform1 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    with open(path, 'rb')as ipath:
        magic, num, rows, cols = struct.unpack('>IIII', ipath.read(16))
        loaded = np.fromfile(path, dtype=np.uint8)
        train_images = loaded[16:].reshape(numb, 784).astype(np.uint8)
        train_images = train_images.reshape(numb, 28, 28)
        train_images = torch.from_numpy(train_images)

        return train_images
train_loader = torch.utils.data.DataLoader(MyDataset(train=True),
                                           batch_size=32, shuffle=True)