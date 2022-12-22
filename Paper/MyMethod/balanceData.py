#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import binascii
import os
import unittest
from Paper.MyMethod.payloadUtil import Mac2Label
from Paper.MyMethod.payloadUtil import rareDevice

maxSampleNum = 10000
data = []
label = []
DataAugFile = r"D:\Documents\shj\胡伟业我的\iie\PytorchProject\Pytorch\data/augment_data.csv"
LabelAugFile = r"D:\Documents\shj\胡伟业我的\iie\PytorchProject\Pytorch\data/augment_label.csv"


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
    if Mac2Label[deviceMac] not in rareDevice:  # 只对稀有样本进行重采样
        return res, ""
    if label.count(Mac2Label[deviceMac]) > maxSampleNum:
        return res, ""
    packetHeader = 0  # 指向每个packet Record首字节
    while packetHeader < len(hexst):
        try:
            i = packetHeader  # 现在i指向packet header的首字节
            i += (16 * 2)  # 跳过Packet Header，现在i指向packet的首字节
            i += (14 * 2)  # 跳过以太网帧格式，不要Mac地址了，现在i指向ip层的首字节
            packetLen = int(hexst[packetHeader + 8 * 2:(packetHeader + 8 * 2) + 4 * 2], 16)  # packet的实际长度，以字节为单位
            packetData = [int(hexst[j:j + 2], 16) for j in
                          range(i, i + min(packetLen * 2 - 14 * 2, embeddingSize * 2), 2)]  # 截取当前数据包指定长度
            # 对当前数据包隐藏ip地址
            for indexIP in range(12, min(12 + 8, len(packetData))):
                 packetData[indexIP] = 0
            if len(packetData) < embeddingSize:  # 当前数据包不够embeddingSize字节
                for i in range(0, embeddingSize - len(packetData)):
                    packetData.append(0)  # 当前packet不足embedding的补上若干个0
            res.append(packetData)  # 添加上当前packet的payload
            packetHeader += (16 * 2 + packetLen * 2)  # 下一个Packet Record记录的起始位置
        except IndexError:
            break  # 发生越界直接break出循环
    if len(res) % seqLen != 0:  # 当前会话的数据包总数不够seqLen切分的，比如总共15个数据包，就需要补一个
        padPacketNum = ((len(res) // seqLen) + 1) * seqLen - len(res)  # 15//8=1,1+1=2,2*8=16,16-15=1，需要补充一个
        for i in range(0, padPacketNum):
            res.append([0] * embeddingSize)
    return res, deviceMac


def agumentData():
    global data
    global label
    paths = [r'../../../USTC-TK2016/3_ProcessedSession/FilteredSession/Train',
             r'../../../USTC-TK2016/3_ProcessedSession/FilteredSession/Test']
    from Paper.MyMethod.payloadUtil import SeqLen
    from Paper.MyMethod.payloadUtil import EmbeddingSize
    for p in paths:
        for d in os.listdir(p):
            for f in os.listdir(os.path.join(p, d)):  # f文件名
                file = os.path.join(p, d, f)  # 一个pcap的完整文件路径
                agumentData, deviceMac = getSessionSample(file, EmbeddingSize, SeqLen)
                if deviceMac == "":
                    continue
                for i in range(0, len(agumentData), SeqLen):  # 假设是16个数据包，SeqLen=8，第一次遍历i=0，第二次遍历i=8
                    data.append(agumentData[i:i + SeqLen])
                    label.append(Mac2Label[deviceMac])
    # 所有文件夹下的所有pcap都解析完了
    data = np.array(data)
    label = np.array(label)
    data = data.reshape(-1, SeqLen * EmbeddingSize)  # 一行是一个embedding
    label = label.reshape(-1, 1)
    print("data's shape=", data.shape)
    print("label's shape=", label.shape)
    np.savetxt(DataAugFile, data, delimiter=',')
    np.savetxt(LabelAugFile, label, delimiter=',')


class TestSklearn(unittest.TestCase):
    def test_utils(self):
        agumentData()

