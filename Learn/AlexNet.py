#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''AlexNet网络的结构-训练-测试文件'''
from __future__ import print_function
from torch import nn
import torch.nn.functional as f
# torch.nn.functional必须在这里import，因为它是个文件名而torch.nn.Module是个类
import numpy as np
import torch.utils.data

class AlexNet(torch.nn.Module):
    '''卷积核全部取原模型的一半，因为只有一个GPU'''
    def __init__(self, num_classes=1000):  # 输出层1000个类别
        super(AlexNet, self).__init__()
        #input[3,227,227]
        '''1：一个卷积池化层'''
        self.feature1 = nn.Sequential(     #nn.Sequential(),一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
            # input[3,227,227],(227-11+2*2)/4+1=56,所以output_feature_map是56*56的
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(11, 11), stride=4, padding=2),# 实际上的AlexNet是有96个卷积核，这里取一半
            # input[48, 56, 56],(56-3)/2+1=27,output[48, 27, 27]
            nn.ReLU(inplace=True),#inplace=True，在原来的数据上改变而不是返回新数据
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        '''2：一个卷积池化层'''
        self.feature2 = nn.Sequential(
            # input[48,27,27],(27-5+2*2)/1+1=27
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=1, padding=2),
            # input[128,27,27],(27-3)/2+1=13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
            # output[128,13,13]
        )
        '''3:一个卷积层'''
        self.feature3 = nn.Sequential(
            # input[128,13,13],(13-3+2*1)/1+1=13
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
            # output[192,13,13]
        )
        '''4:一个卷积层'''
        self.feature4 = nn.Sequential(
            # input[192,13,13],(13-3+2*1)/1+1=13
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
            # output[192,13,13]
        )
        '''5:一个卷积池化层'''
        self.feature5 = nn.Sequential(
            # input[192,13,13],(13-3+2*1)/1+1=13
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            # input[128,13,13],(13-3+0*2)/2+1=6
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
            # output[128,6,6]
        )
        '''6：一个全连接层'''
        self.fullConnect1 = nn.Sequential(
            # intput[128,6,6],打平成一维：一行是一个样本，所以一个样本共有128*6*6=4608个维度，实际上训练的时候数据张量的[0]维是batch_size，所以x=torch.flatten(x, start_dim=1)
            nn.Linear(in_features=128 * 6 * 6, out_features=2048),
            nn.Dropout(p=0.5),  # 在训练的时候每个神经元以0,5的概率停止工作
            nn.ReLU(inplace=True)
        )
        '''7:一个全连接层'''
        self.fullConnect2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2048),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True)
        )
        '''8:输出层'''
        self.OutPut = nn.Sequential(
            nn.Linear(2048, num_classes)#num_classes=1000
        )

    def forward(self, x):
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        x = self.feature4(x)
        x = self.feature5(x)
        x = torch.flatten(x, start_dim=1)  # 将每一个样本展平成一维，或者：x = x.view(x.size()[0],-1)
        # 因为x的第一个维度是批样本个数
        x = self.fullConnect1(x)
        x = self.fullConnect2(x)
        x = self.OutPut(x)
        return x


def train_AlexNet(TrainData:torch.utils.data.DataLoader):
    '''
    训练LeNet5的函数
    :param TrainData: 训练集数据
    :param y_data: 训练集标签
    :param TestData: 测试集数据
    :return:
    '''
    # 超参数
    use_gpu = torch.cuda.is_available()  # torch可以使用GPU时会返回true
    LR = 0.001  # 学习率
    epoch = 50
    model_AlexNet=AlexNet()
    criterion_AlexNet = torch.nn.CrossEntropyLoss()  # 分类常用交叉熵损失
    optimizer_AlexNet = torch.optim.Adam(model_AlexNet.parameters(), lr=LR, betas=(0.9, 0.999))  # 后两项是默认值
    if use_gpu:
        model_AlexNet = model_AlexNet.cuda()  # 把网络模型放到gpu上
    else:
        pass  # 不显式指定的话默认使用cpu

    model_AlexNet.train()  # 训练CNN一定要调用.train()，测试的时候调用.eval()

    torch.manual_seed(2)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。
    for e in range(epoch):  # 每一次迭代都是针对全体数据做的
        sum_loss = 0.0
        for batch_index, (data, label) in enumerate(TrainData):  # 批数据的下标，该批数据的data及对应的label
            if use_gpu:
                data, label = data.cuda(), label.cuda()  # 把数据放到GPU上
            else:
                pass
            label_pre = model_AlexNet(data)
            loss = criterion_AlexNet(label_pre, label)
            optimizer_AlexNet.zero_grad()
            loss.backward()
            optimizer_AlexNet.step()  # 更新网络参数
            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if batch_index % 100 == 99:
                print('[%d, %d] loss: %.03f'
                      % (e + 1, batch_index + 1, sum_loss / 100))  # 在第e次迭代第100个batch时的平均误差
                sum_loss = 0.0

if __name__ == '__main__':
    import torchvision as tv
    normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = tv.datasets.ImageFolder('../data/image',
        tv.transforms.Compose([
                tv.transforms.RandomResizedCrop(227), #AlexNet输入
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                normalize
        ]))
    # 使用dataloader迭代器来加载数据集
    my_batch_size = 256  # 一次训练的样本个数
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=my_batch_size,#数据集划分批次
                                               num_workers=2,
                                               pin_memory=True
                                               )
    '''
    #打印训练集的第一个图片
    plt.figure()
    img=tv.utils.make_grid(train_dataset.data[0]).numpy()#取训练集的data中的第一张图片
    plt.imshow(np.transpose(img,(1,2,0)))
    plt.show()
    '''
    train_AlexNet(train_loader)

'''
卷积与池化输出维度计算公式：
    https://www.cnblogs.com/aaronhoo/p/12464941.html
'''