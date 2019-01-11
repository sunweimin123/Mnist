# coding:utf-8
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import preprocessing
import torch.nn as nn
import get_mnist_data
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import datasets, transforms

train_images, train_labels, test_images, test_labels = get_mnist_data.load_mnist()


BATCH_SIZE = 60

label = preprocessing.LabelEncoder()
one_hot = preprocessing.OneHotEncoder(sparse = False)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

net = Net()







optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = nn.CrossEntropyLoss()




torch_dataset = Data.TensorDataset(train_images, train_labels)

loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据

)


for t in range(10):
    train_loss = 0.
    train_acc = 0.
    for step, (batch_x, batch_y) in enumerate(loader):
        data, target = Variable(batch_x), Variable(batch_y)
        optimizer.zero_grad()  # 优化器梯度初始化为零
        output = net(data)  # 把数据输入网络并得到输出，即进行前向传播

        target =target.reshape(BATCH_SIZE)
        loss = F.nll_loss(output, target)  # 计算损失函数
        train_loss += loss.data
        pred = torch.max(output, 1)[1]
        train_correct = (target == pred).sum()
        train_acc += train_correct.data
        loss.backward()  # 反向传播梯度
        optimizer.step()
        if step % 10 == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                step, step * len(data), len(loader.dataset),
                       100. * step / len(loader), loss.data))
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
             train_images)), train_acc / (len(train_images))))
    #     #batch_x = batch_x.reshape(BATCH_SIZE,1,28,28)
    #
    #
    #     data, target = Variable(batch_x), Variable(batch_y)
    #     prediction = net(batch_x)
    #
    #
    #     loss = loss_func(prediction,batch_y.long().squeeze())
    #     train_loss += loss.data
    #     pred = torch.max(prediction, 1)[1]
    #     train_correct = (pred == batch_y.long()).sum()
    #
    #     train_acc += train_correct.data
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     # if step % 10 == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
    #     #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #     #         10, step * len(data), len(loader.dataset),
    #     #                100. * step / len(loader), loss.data))
    #
    # # print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
    # #     train_images)), train_acc / (len(train_images))))


torch.save(net.state_dict(),'net.pkl')




# net.load_state_dict(torch.load('net.pkl'))
# test_output = net(test_images[:20].reshape(20,1,28,28))
# print test_images[:20][1].reshape(28,28)
# print test_images[:20][2].reshape(28,28)
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_labels[:20].numpy(), 'real number')

