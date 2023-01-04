#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''模型和参数选型'''
import torch
import numpy as np
from Paper import train_eval
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#小米设备的标签列表
classList = ["mitu_story","xiaobai_camera","xiaomi_control_plug"]
from SHJ.models.BiLSTM import Config
from SHJ.models.BiLSTM import Model
from SHJ.genPayloadData import SeqLen
from SHJ.genPayloadData import LabelNum
from SHJ.genPayloadData import EmbeddingSize
from SHJ.genPayloadData import DataFilePath
from SHJ.genPayloadData import LabelFilePath
if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(2)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定
    torch.cuda.manual_seed_all(1)  # 为所有GPU设置随机初始化种子
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    config = Config(SeqLen=SeqLen, LabelNum=LabelNum, EmbeddingSize=EmbeddingSize, Epoch=20, classList=classList)
    model = Model(config)  # 网络模型

    seq_len = config.seq_len  # 一个句子的长度
    n_features = config.num_feature  # 最终选取的特征的个数，embedding的维数
    num_labels = config.num_labels  # 类别数目

    df_data = pd.read_csv(DataFilePath, header=None, sep=',', dtype=np.float32)
    df_label = pd.read_csv(LabelFilePath, header=None, sep=',', dtype=np.uint8)
    data = df_data.values
    label= df_label.values

    print("sample num=", label.size)

    # 对于原始的三维数据，要归一化必须展成二维,所以从csv文件里读取,文件每seq_len行是一个样本，所以csv的行数必须能被seq_len整除
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(data)
    data = scaler.transform(data)

    data = data.reshape(-1, seq_len, n_features)
    label = label.reshape(label.size)

    # 8:1:1划分训练集 验证集 测试集
    X, test_data, Y, test_label = train_test_split(data, label, test_size=0.1, random_state=1)
    train_data, valid_data, train_label, valid_label = train_test_split(X, Y, test_size=0.125, random_state=1)

    train_data = torch.FloatTensor(train_data)  # 训练数据必须是float
    train_label = torch.LongTensor(train_label)  # 标签

    valid_data = torch.FloatTensor(valid_data)
    valid_label = torch.LongTensor(valid_label)

    test_data = torch.FloatTensor(test_data)
    test_label = torch.LongTensor(test_label)

    Trainloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(train_data, train_label),  # 把训练集和其标签组合到一起
        batch_size=config.batch_size,  # batch size
        shuffle=True,  # 要不要打乱数据
    )
    ValidLoader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(valid_data, valid_label),  # 把验证集和其标签组合到一起
        batch_size=config.batch_size,  # batch size
        shuffle=True,  # 要不要打乱数据
    )

    train_eval.train(Trainloader, ValidLoader, config, model)  # 训练

    Testloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(test_data, test_label),
        batch_size=128,  # batch size
        shuffle=True,  # 要不要打乱数据
    )

    train_eval.test(config, model, Testloader)