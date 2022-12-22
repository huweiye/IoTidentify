#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import unittest

import torch

import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch.utils.data
import matplotlib.pyplot as plt
import pandas as pd



class TestSklearn(unittest.TestCase):
    def test_IoTAndNonIoT(self):
        '''划分物联网设备和非物联网设备'''
        from SHJ import IoTANDNonIoT
        def loadDataSet():
            if not os.path.exists(IoTANDNonIoT.DataPath):#只在csv数据集文件不存在时读取pcap生成数据集csv文件
                IoTANDNonIoT.GenData(IoTANDNonIoT.FileDir)
            df_data = pd.read_csv(IoTANDNonIoT.DataPath, header=None, sep=',', dtype=np.float32)
            df_label = pd.read_csv(IoTANDNonIoT.LabelPath, header=None, sep=',', dtype=np.uint8)
            data = df_data.values
            label = df_label.values
            return data,label
        data,label=loadDataSet()#加载数据集
        IoTANDNonIoT.ClassifyIoTAndNonIoT(data, label)

    def test_filterXiaomi(self):
        dir=r"D:\Documents\shj\DataSet\IoT identification\Aisia CCS2020-Your smart home can't keep a secret-包含小米设备\iot-traffic-dataset-20221212T071426Z-001\iot-traffic-dataset\xiaomi"
        from SHJ import FilterXiaomi
        FilterXiaomi.printXiaomiMac(dir)