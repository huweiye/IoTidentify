# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, SeqLen,LabelNum,EmbeddingSize,Epoch,classList):
        self.batch_size = 128
        self.num_feature = EmbeddingSize  # 词向量的维数，LSTM的第一个参数值，对dport进行embedding后是30
        self.num_labels = LabelNum  # 类别数目
        self.hidden_size = 64  # LSTM的隐藏层维度，论文中取64
        self.num_layers = 2  # LSTM的深度,（在竖直方向堆叠的多个LSTM单元的层数）,论文中为1
        self.dropout = 0.5  # 训练时神经元随机失活概率，论文中为0.5
        self.learn_rate = 0.001  # 学习率，论文中为0.001
        self.seq_len = SeqLen  # 一个完整句子的长度，默认为10
        self.save_path = r"./BiLSTM_Att.ckpt"  # 模型训练结果保存位置
        self.epoch = Epoch
        self.class_list = classList
        self.linear_hidden_size = 64


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, config:Config):
        super(Model, self).__init__()
        self.conf = config
        self.lstm = nn.LSTM(input_size=config.num_feature, hidden_size=config.hidden_size, num_layers=config.num_layers,bidirectional=True, batch_first=True)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.linear_hidden_size)
        self.fc = nn.Linear(config.linear_hidden_size, config.num_labels)

    def forward(self, x):
        H, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 10, 128]
        M = self.tanh1(H)  # [128, 10, 128]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 10, 1]
        out = H * alpha  # [128, 10, 128]
        out = torch.sum(out, 1)  # [128, 128]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 21]
        return out
