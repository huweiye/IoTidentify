#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy
import numpy as np
import binascii
import os
import unittest

Mac2Label = {
    "d0:52:a8:00:67:5e": 0,  # Smart Things
    "44:65:0d:56:cc:d3": 1,  # Amazon Echo
    "70:ee:50:18:34:43": 2,  # Netatmo Welcome
    "f4:f2:6d:93:51:f1": 3,  # TP-Link Day Night Cloud camera
    "00:16:6c:ab:6b:88": 4,  # Samsung SmartCam
    "30:8c:fb:2f:e4:b2": 5,  # Dropcam
    "00:62:6e:51:27:2e": 6,  # Insteon Camera
    "00:24:e4:11:18:a8": 7,  # Withings Smart Baby Monitor
    "ec:1a:59:79:f4:89": 8,  # Belkin Wemo switch
    "50:c7:bf:00:56:39": 9,  # TP-Link Smart plug
    "74:c6:3b:29:d7:1d": 10,  # iHome
    "ec:1a:59:83:28:11": 11,  # Belkin wemo motion sensor
    #"18:b4:30:25:be:e4": 12,  # NEST Protect smoke alarm
    "70:ee:50:03:b8:ac": 12,  # Netatmo weather station
    "00:24:e4:1b:6f:96": 13,  # Withings Smart scale
    #"74:6a:89:00:2e:25": 15,  # Blipcare Blood Pressure meter
    "00:24:e4:20:28:c6": 14,  # Withings Aura smart sleep sensor
    "d0:73:d5:01:83:08": 15,  # Light Bulbs LiFX Smart Bulb
    "18:b7:9e:02:20:44": 16,  # Triby Speaker
    "e0:76:d0:33:bb:85": 17,  # PIX-STAR Photo-frame
    "70:5a:0f:e4:9b:c0": 18,  # HP Printer
    "08:21:ef:3b:fc:e3": 19,  # Samsung Galaxy Tab
}
LanMac = "14:cc:20:51:33:ea"
SeqLen = 10  # 灰度图的行数
EmbeddingSize = 120  # 灰度图的列数，SeqLen*EmbeddingSize需要等于2_Process2Session.ps1文件里的截取长度
LabelNum = len(Mac2Label)
OriginDataFilePath = r'D:\Documents\shj\胡伟业我的\iie\PytorchProject\Pytorch\data/origin_data-X.csv'
OriginLabelFilePath = r'D:\Documents\shj\胡伟业我的\iie\PytorchProject\Pytorch\data/origin_label-X.csv'
rareDevice = [5, 9, 10, 13, 16, 18]  # 所有原来的样本数小于百分之一的样本
denseDevice = [1, 11]

# def getSessionSample(filename, width):
#     with open(filename, 'rb') as f:
#         content = f.read()
#     hexst = binascii.hexlify(content)
#     if filename.endswith(".pcap"):#读的是ALLLayers的pcap文件，需要偏移pcap文件头
#         #参考链接：https://blog.csdn.net/lu_embedded/article/details/124952413
#         #Pcap 文件全局报头的长度为 24 字节
#         deviceMac=getDeviceMac(hexst[24*2:])
#     fh = numpy.array([int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)])
#     rn = len(fh) // width
#     fh = numpy.reshape(fh[:rn * width], (-1, width))
#     fh = numpy.uint8(fh)
#     return fh,deviceMac


def getSessionSample(filename, embeddingSize: int, seqLen: int):  # embeddingSize：一个数据包截取的长度，以字节为单位;seqLen:截取多少个数据包
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    # Pcap 文件全局报头的长度为 24 字节
    hexst = hexst[24 * 2:]
    res = []  # 从流的前SeqLen个数据包生成的矩阵，SeqLen*embeddingSize
    deviceMac = getDeviceMac(hexst)  # 获取设备Mac地址
    if deviceMac == "":
        return res, deviceMac
    # print(hexst)
    packetHeader = 0  # 指向每个packet Record首字节

    while packetHeader < len(hexst) and len(res) < seqLen:
        try:
            i = packetHeader  # 现在i指向packet header的首字节
            i += (16 * 2)  # 跳过Packet Header，现在i指向packet的首字节
            i += (14 * 2)  # 跳过以太网帧格式，不要Mac地址了，现在i指向ip层的首字节
            packetLen = int(hexst[packetHeader + 8 * 2:(packetHeader + 8 * 2) + 4 * 2], 16)  # packet的实际长度，以字节为单位
            packetData = [int(hexst[i:i + 2], 16) for i in
                          range(i, i + min(packetLen * 2 - 14 * 2, embeddingSize * 2), 2)]  # 截取当前数据包指定长度
            # 对当前数据包隐藏ip地址
            for indexIP in range(12, min(12 + 8, len(packetData))):
                packetData[indexIP] = 0
            if len(packetData) < embeddingSize:
                for i in range(0, embeddingSize - len(packetData)):
                    packetData.append(0)  # 当前packet不足embedding的补上若干个0
            res.append(packetData)  # 添加上当前packet的payload
            packetHeader += (16 * 2 + packetLen * 2)  # 下一个Packet Record记录的起始位置
        except IndexError:
            break  # 发生越界直接break出循环
    if len(res) < seqLen:  # 当前会话不足seqLen个包，则补充0
        for i in range(0, seqLen - len(res)):
            res.append([0] * embeddingSize)
    return res, deviceMac


def getDeviceMac(packetRecordList):  # 找到十六进制文件里的Iot设备mac，如果不存在的设备则为""
    # 对pcap文件，每个数据包的捕获记录Packet Record都由Packet Header + Packet Data组成
    packetData = packetRecordList[16 * 2:]  # Packet Header 长度固定为 16 字节,Packet Header之后才是第一个数据包的Data
    # 通过找第一个数据包的srcMac和dstMac确定Label
    dstMacBytes = packetData[:6 * 2]
    srcMacBytes = packetData[6 * 2:6 * 2 + 6 * 2]
    dstMac = ""
    for i in range(0, len(dstMacBytes), 2):
        dstMac += str(dstMacBytes[i:i + 2], 'UTF-8')
        dstMac += ":"
    dstMac = dstMac[:-1]
    srcMac = ""
    for i in range(0, len(srcMacBytes), 2):
        srcMac += str(srcMacBytes[i:i + 2], 'UTF-8')
        srcMac += ":"
    srcMac = srcMac[:-1]
    deviceMac = ""
    if srcMac in Mac2Label.keys():
        deviceMac = srcMac
    elif dstMac in Mac2Label.keys():
        deviceMac = dstMac
    return deviceMac


def genData():
    data = []
    label = []
    paths = [r'../../../USTC-TK2016/3_ProcessedSession/FilteredSession/Train',
             r'../../../USTC-TK2016/3_ProcessedSession/FilteredSession/Test']
    for p in paths:
        for d in os.listdir(p):
            for f in os.listdir(os.path.join(p, d)):  # f文件名
                file = os.path.join(p, d, f)  # 一个pcap的完整文件路径
                Sessionmatrix, deviceMac = getSessionSample(file, EmbeddingSize, SeqLen)
                if deviceMac == "":
                    continue
                data.append(Sessionmatrix)
                label.append(Mac2Label[deviceMac])
    # 所有文件夹下的所有pcap都解析完了
    data = np.array(data)
    label = np.array(label)
    data = data.reshape(-1, SeqLen * EmbeddingSize)  # 一行是一个embedding
    label = label.reshape(-1, 1)
    print("data's shape=", data.shape)
    print("label's shape=", label.shape)
    np.savetxt(OriginDataFilePath, data, delimiter=',')
    np.savetxt(OriginLabelFilePath, label, delimiter=',')


class TestSklearn(unittest.TestCase):
    def test_utils(self):
        genData()

    def test_sample(self):
        '''
        打印样本情况，用于说明样本分布不均衡的问题
        :return:
        '''
        def pieChart():
            import pandas as pd
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
            plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题
            df_label = pd.read_csv(
                OriginLabelFilePath,
                header=None, sep=',', dtype=np.uint8)
            label = df_label.values
            sampleSum = label.size
            print("sample num={}".format(sampleSum))
            sampleNumList = []
            for l in range(LabelNum):
                sampleNumList.append(np.sum(label == l))
                print("Label={} sample num={} pro={:.2%}".format(l, sampleNumList[-1],sampleNumList[-1] / sampleSum))
                if sampleNumList[-1] / sampleSum < 0.01:  # 样本占比小于百分之一的认为是小样本
                    print("label {} is rare sample".format(l))
            sampleNumList=[]
            explored=[0]*LabelNum
            for r in rareDevice:
                explored[r]=0.3
            from Paper.run import classList
            plt.pie(sampleNumList, labels=classList, autopct='%.2f%%', pctdistance=0.6,explode=explored)
            plt.title('Iot设备样本数目饼图')  # 加标题
            plt.show()
        pieChart()