#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as f
# torch.nn.functional必须在这里import，因为是个文件名而torch.nn.Module是个类
import numpy as np
import torch.utils.data

class Config(object):
    def __init__(self,SeqLen,LabelNum,EmbeddingSize,Epoch,classList):
        self.batch_size = 128
        self.num_feature = EmbeddingSize  # 词向量的维数，LSTM的第一个参数值
        self.num_labels = LabelNum#类别数目
        self.hidden_size = 64  # LSTM的隐藏层维度，论文中取64
        self.num_layers = 2  # LSTM的深度,（在竖直方向堆叠的多个LSTM单元的层数）,论文中为1
        self.dropout = 0.5  # 训练时神经元随机失活概率，论文中为0.5
        self.learn_rate=0.001#学习率，论文中为0.001
        self.seq_len = SeqLen  # 一个完整句子的长度，默认为10
        self.save_path =r"./CNNBiLSTM.ckpt" # 模型训练结果保存位置
        self.epoch=Epoch
        self.class_list=classList


class Model(torch.nn.Module):
    def __init__(self, config:Config):
        self.conf=config
        super(Model, self).__init__()
        '''Bi-LSTM'''
        # origin shape=[batch_size,seq_len,n_feature]=[128,10,200]
        self.rnn = torch.nn.LSTM(input_size=config.num_feature, hidden_size=config.hidden_size, num_layers=config.num_layers, bidirectional=True, batch_first=True)
        #rnn output shape=[batch_size,seq_len,hidden_size]=[128,10,128]

        '''第一个卷积池化层'''
        self.feature1 = torch.nn.Sequential(
            # 输入[128,1,10,128]，一张图片10*128，
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 10), padding=2, stride=(1,2)),
            #图片高=(10-2+2*2)/1+1=13，图片宽=(128-10+2*2)/2+1=62
            torch.nn.ReLU(inplace=True),
            # [batch_size,channels,feature_width,feature_length]=[128,32,13,62]
            torch.nn.MaxPool2d(kernel_size=(3, 12), stride=(2,2))
            # 经filter=(2,2),stride=2的池化，输出的每个特征图的高=(13-3+0*2)/2+1=6，宽=（62-12+0*2）/2+1=26
        )
        #feature1 output shape=[batch_size,channels,height,width]=[128,32,6,26]
        '''第二个卷积池化层'''
        self.feature2 = torch.nn.Sequential(
            # input[128,32,6,26],高=(6-2+2*2)/1+1=9，宽=(26-5+2*2)/1+1=26
            torch.nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(2, 5), stride=1, padding=2),
            # [128,96,9,26]
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(1, 4), stride=2)#(9-1)/2+1=5,(26-4)/2+1=12
            # output[128,96,5,12]
        )
        #feature2 output shape=[128,96,5,12]
        '''一个全连接层'''
        self.fc1=torch.nn.Sequential(
            #forward()里先将96通道的二维图片打平成一维向量
            torch.nn.Linear(in_features=96*5*12, out_features=128),
            torch.nn.Dropout(p=config.dropout),  # dropout都被放到全连接层之后，鲜有放到卷积层之后
            torch.nn.ReLU(inplace=True)
        )
        '''第二个全连接层'''
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.Dropout(p=config.dropout),  # dropout都被放到全连接层之后，鲜有放到卷积层之后
            torch.nn.ReLU(inplace=True)
        )
        '''输出'''
        self.output=torch.nn.Linear(in_features=64, out_features=config.num_labels)

    def forward(self, x):
        x, _ = self.rnn(x)# output=[batch_size, seq_len, hidden_size * num_direction]=[128, 10, 128]
        x = x.unsqueeze(1)  # 扩充成单通道图像,[128,1,10,128]
        x=self.feature1(x) #[batch_size,channels,height,width]=[128,32,6,26]
        x=self.feature2(x)#[128,96,5,12]
        x = x.reshape(x.size()[0], -1)  # 重新塑形,将多维数据重新塑造为二维数据
        x = self.fc1(x)
        x=self.fc2(x)
        output = self.output(x)
        return output