from __future__ import print_function
import torch
import torch.nn.functional as f
# torch.nn.functional必须在这里import，因为是个文件名而torch.nn.Module是个类
import numpy as np
import torch.utils.data

class Config(object):
    """配置参数"""
    def __init__(self):
        self.batch_size = 128
        self.num_feature = 11  # 词向量的维数，LSTM的第一个参数值，对dport进行embedding后是30
        self.num_labels = 11#类别数目
        self.hidden_size = 64  # LSTM的隐藏层维度，论文中取64
        self.num_layers = 1  # LSTM的深度,（在竖直方向堆叠的多个LSTM单元的层数）,论文中为1
        self.dropout = 0.5  # 训练时神经元随机失活概率，论文中为0.5
        self.learn_rate=0.001#学习率，论文中为0.001
        self.seq_len = 100  # 一个完整句子的长度，论文中traffic window取100
        self.save_path =r"./AsiaCCS.ckpt" # 模型训练结果保存位置

class AsiaCCS2020(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_layer, n_output,batch_size,seq_len):
        '''
        :param n_features: 特征维数，即一个word embedding的元素个数
        :param n_hidden:每个LSTM单元在某时刻t的输出的ht的维度
        :param n_layer:RNN层的个数：（在竖直方向堆叠的多个LSTM单元的层数）
        :param n_output:最后的全连接层的输出
        '''
        super(AsiaCCS2020, self).__init__()
        self.embedding=torch.nn.Sequential( #Embedding层，将dport这一离散特征转换成19维embedding


        )
        self.lstm1 = torch.nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layer,bidirectional=True,batch_first=True)
        # batch_first默认为False，但是通常应该置为True，这样数据维度的第0维就是表示样本，大小就是batch_size
        self.fc1 = torch.nn.Linear(in_features=n_hidden*seq_len, out_features=n_output)#线性层的输入是二维张量，其中第0维度一定是样本，第1维度才是每个样本的特征数

        self.batch_size=batch_size

    def forward(self, x):
        '''
        原始数据的形状=[batch_size,seq_len,n_features]
        输入数据的batch_size=3,seq_len=5,n_feature=10
        '''
        print("origin data's shape=",x.size())#--------------------[3,5,10]

        x, (hn, cn) = self.lstm1(x)
        '''
        lstm的输出的形状=[batch_size,seq_len,n_hidden]
        '''
        print("output size() of lstm1=",x.size())#---------------------[3,5,20]

        #print("before shape",x)
        x=x.reshape(self.batch_size,-1)#Linear的输入必须是二维张量,那就是一个样本，它的特征向量是：每个时间步的embedding的拼接
        #print(x)

        '''
        input shape of Linear=[batch_size,seq_len*n_hidden]
        '''
        print("input data's size() of Linear=", x.size())  # ---------------------[3,5*20]
        output=f.softmax(self.fc1(x),dim=1)

        return output