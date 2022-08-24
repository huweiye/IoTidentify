#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as f
# torch.nn.functional必须在这里import，因为是个文件名而torch.nn.Module是个类
import numpy as np
import torch.utils.data

class Config(object):
    """配置参数"""
    def __init__(self,SeqLen,LabelNum,EmbeddingSize,Epoch,classList):
        self.batch_size = 128
        self.num_feature = EmbeddingSize  # 词向量的维数
        self.num_labels = LabelNum#类别数目
        self.hidden_size = 64  # LSTM的隐藏层维度，论文中取64
        self.num_layers = 2  # LSTM的深度,（在竖直方向堆叠的多个LSTM单元的层数）,论文中为1
        self.dropout = 0.5  # 训练时神经元随机失活概率，论文中为0.5
        self.learn_rate=0.001#学习率，论文中为0.001
        self.seq_len = SeqLen  # 一个完整句子的长度，默认为10
        self.save_path =r"./BiLSTM.ckpt" # 模型训练结果保存位置
        self.epoch=Epoch
        self.class_list=classList

class Model(torch.nn.Module):
    def __init__(self, config:Config):
        '''
        :param num_feature: 特征维数，即一个word embedding的元素个数
        :param hidden_size:每个LSTM单元在某时刻t的输出的ht的维度
        :param num_layers:RNN层的个数：（在竖直方向堆叠的多个LSTM单元的层数）
        :param num_labels:最后的全连接层的输出
        '''
        self.conf=config
        super(Model, self).__init__()
        self.embeded=torch.nn.Sequential(
            #Embedding层，将dport这一离散特征转换成19维embedding，暂时忽略
        )
        self.rnn = torch.nn.LSTM(input_size=config.num_feature, hidden_size=config.hidden_size, num_layers=config.num_layers, bidirectional=True, batch_first=True)
        self.fc=torch.nn.Sequential(
            torch.nn.Dropout(p=config.dropout),
            #如果只取最后一个时间步则in_features=config.hidden_size*2
            torch.nn.Linear(in_features=config.hidden_size*2*config.seq_len, out_features=64),#线性层的输入是二维张量，其中第0维度一定是样本，第1维度才是每个样本的特征数
            torch.nn.ReLU(inplace=True)
        )
        self.output=torch.nn.Linear(in_features=64, out_features=config.num_labels)

    def forward(self, x):
        rnnOutput, _ = self.rnn(x)# [batch_size, seq_len, hidden_size * num_direction]=[128, 10, 128]


        linearInput=rnnOutput.reshape(rnnOutput.shape[0],-1)#取全部时间步
        #linearInput=rnnOutput[:,-1,:]#只取句子最后时刻的 hidden state

        linearOutput=self.fc(linearInput)

        output=self.output(linearOutput)

        return output