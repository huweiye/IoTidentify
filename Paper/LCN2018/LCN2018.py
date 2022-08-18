import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f

'''
title:Automatic Device Classification from Network Traffic Streams of Internet of Things
'''


class Config(object):
    """配置参数"""

    def __init__(self, n_feature, seq_len, n_labels):
        '''
        :param n_feature: 一个embedding的维数，对应论文里就是向量x的维数
        :param seq_len: 一个句子的维数，对应论文里就是sub[dj,iT~(i+1)T]里packet的个数，在代码里就是输入lstm的时间步个数seq_len，等于lstm的output.size()[1]
                        但是原论文里是在sub上滑动size=t的窗口，输入lstm的seq_len也是t
        '''
        self.batch_size = 128
        self.num_feature = n_feature  # 词向量的维数，LSTM的第一个参数值
        self.num_labels = n_labels
        self.hidden_size = 6  # LSTM的隐藏层维度
        self.num_layers = 2  # LSTM的深度
        self.dropout = 0.6  # 训练时神经元随机失活概率
        self.learn_rate=0.05#论文里设置的
        self.seq_len = seq_len  # 一个完整句子的长度
        self.num_filter = 32  # 卷积层使用滤波器的个数
        self.filter_size = 2  # 卷积层滤波器的大小
        self.save_path =r"/Users/bytedance/Documents/胡伟业我的/iie/PytorchProject/Pytorch/Paper/LCN2018/LCN2018.ckpt" # 模型训练结果


class LCN2018(nn.Module):
    def __init__(self, config):
        super(LCN2018, self).__init__()
        self.config = config
        #origin data shape=[batch_size,seq_len,n_feature]=[128,6,6]
        self.lstm = nn.LSTM(config.num_feature, config.hidden_size, config.num_layers,
                            bidirectional=False, batch_first=True, dropout=config.dropout)
        # output of lstm=[batch_size,seq_len,hidden_size]=[128,6,6]
        self.feature1 = nn.Sequential(
            # 输入单通道,输出是32个feature map，因为卷积核是2，padding=2，所以输出的每个特征图的大小=((6-2+2*2)/1)+1=9
            nn.Conv2d(1, config.num_filter, (config.filter_size, config.filter_size), padding=2, stride=1),
            nn.ReLU(inplace=True),  # inplace=True，在原来的数据上改变而不是返回新数据
            # [batch_size,channels,feature_width,feature_length]=[128,32,9,9]
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            # 经filter=(2,2),stride=2的池化，输出的每个特征图的大小=((9-2)/2)+1=4
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features=config.num_filter * 4 * 4, out_features=128),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Linear(64, config.num_labels)

    def forward(self, x):
        x, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size]=[128, 6, 6]
        x = x.unsqueeze(1)  # 扩充成单通道图像,[128,1,6,6]
        x=self.feature1(x)
        x = x.reshape(x.size()[0], -1)  # 重新塑形,将多维数据重新塑造为二维数据
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return out
