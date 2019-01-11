# coding:utf-8
import os
import struct
import numpy as np
import torchvision.transforms as transforms
import torch

def load_mnist():
    transform1 = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    root = 'mnist_dataset'

    train_labels_path = os.path.join(root,'train-labels.idx1-ubyte')
    train_images_path = os.path.join(root,'train-images.idx3-ubyte')

    test_labels_path = os.path.join(root,'t10k-labels.idx1-ubyte')
    test_images_path = os.path.join(root,'t10k-images.idx3-ubyte')

    with open(train_labels_path,'rb')as lpath:
        magic,n = struct.unpack('>II',lpath.read(8))
        train_labels = np.fromfile(lpath, dtype=np.uint8).astype(np.int64)

        train_labels = torch.from_numpy(train_labels)
    with open(train_images_path,'rb')as ipath:
        magic,num,rows,cols = struct.unpack('>IIII', ipath.read(16))
        loaded = np.fromfile(train_images_path, dtype=np.uint8)
        train_images = loaded[16:].reshape(len(train_labels), 784).astype(np.float32)
        train_images = train_images.reshape(60000,1,28,28)
        tem = np.transpose(train_images, (0,2, 3, 1))
        for i in range(60000):
            train_images[i] = transform1(tem[i])
        train_images = torch.from_numpy(train_images)


    with open(test_labels_path, 'rb') as lpath:
        magic, n = struct.unpack('>II', lpath.read(8))
        test_labels = np.fromfile(lpath, dtype=np.uint8).astype(np.float)
        test_labels = torch.from_numpy(test_labels)
    with open(test_images_path, 'rb') as ipath:
        magic, num, rows, cols = struct.unpack('>IIII', ipath.read(16))
        loaded = np.fromfile(test_images_path, dtype=np.uint8)
        test_images = loaded[16:].reshape(len(test_labels), 784).astype(np.float)
        test_images = test_images.reshape(10000, 1, 28, 28)
        tem = np.transpose(test_images, (0, 2, 3, 1))
        for i in range(10000):
            test_images[i] = transform1(tem[i])
        test_images = torch.from_numpy(test_images)


    return train_images, train_labels, test_images, test_labels





train_images, train_labels, test_images, test_labels = load_mnist()

# plt.imshow(train_images[5000].reshape(28,28), cmap=plt.cm.gray)
# plt.show()
# print train_images[1].reshape(28,28)



# transform1 = transforms.Compose([transforms.ToTensor()])
#
# d1 = [1,2,3,4,5,6]
# d2 = [4,5,6,7,8,9]
# d3 = [7,8,9,10,11,14]
# d4 = [11,12,13,14,15,15]
# d5 = [d1,d2,d3,d4]
# d = np.array([d5,d5,d5],dtype=np.float32)
# d_t = np.transpose(d,(1,2,0))
# d_t_trans = transform1(d_t)
#
# print transform1(d_t).float().div(255)


