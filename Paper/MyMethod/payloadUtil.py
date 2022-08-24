#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy
import numpy as np
import binascii
import os
import unittest

Mac2Label={
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
    "18:b4:30:25:be:e4": 12,  # NEST Protect smoke alarm
    "70:ee:50:03:b8:ac": 13,  # Netatmo weather station
    "00:24:e4:1b:6f:96": 14,  # Withings Smart scale
    #"74:6a:89:00:2e:25": 15,  # Blipcare Blood Pressure meter
    "00:24:e4:20:28:c6": 15,  # Withings Aura smart sleep sensor
    "d0:73:d5:01:83:08": 16,  # Light Bulbs LiFX Smart Bulb
    "18:b7:9e:02:20:44": 17,  # Triby Speaker
    "e0:76:d0:33:bb:85": 18,  # PIX-STAR Photo-frame
    "70:5a:0f:e4:9b:c0": 19,  # HP Printer
    "08:21:ef:3b:fc:e3":20 ,#Samsung Galaxy Tab
}
LanMac = "14:cc:20:51:33:ea"
SeqLen=8 #灰度图的行数
EmbeddingSize=128#灰度图的列数，SeqLen*EmbeddingSize需要等于2_Process2Session.ps1文件里的截取长度
LabelNum=len(Mac2Label)
DataFilePath=r'D:\Documents\shj\胡伟业我的\iie\PytorchProject\Pytorch\data/TMC2018_payload_data.csv'
LabelFilePath=r'D:\Documents\shj\胡伟业我的\iie\PytorchProject\Pytorch\data/TMC2018_payload_label.csv'


def getMatrixfrom_pcap(filename, width):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    if filename.endswith(".pcap"):#读的是pcap文件，需要偏移pcap文件头
        #参考链接：https://blog.csdn.net/lu_embedded/article/details/124952413
        #Pcap 文件全局报头的长度为 24 字节
        deviceMac=getDeviceMac(hexst[24*2:])
    fh = numpy.array([int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)])
    rn = len(fh) // width
    fh = numpy.reshape(fh[:rn * width], (-1, width))
    fh = numpy.uint8(fh)
    return fh,deviceMac

def getDeviceMac(packetRecordList):#找到十六进制文件里的Iot设备mac，如果不存在的设备则为""
    # 对pcap文件，每个数据包的捕获记录Packet Record都由Packet Header + Packet Data组成
    packetData = packetRecordList[16 * 2:]  # Packet Header 长度固定为 16 字节,Packet Header之后才是第一个数据包的Data
    # 通过找第一个数据包的srcMac和dstMac确定Label
    dstMacBytes=packetData[:6*2]
    srcMacBytes=packetData[6*2:(6+6)*2]
    dstMac=""
    for i in range(0, len(dstMacBytes), 2):
        dstMac+=str(dstMacBytes[i:i + 2],'UTF-8')
        dstMac+=":"
    dstMac=dstMac[:-1]
    srcMac=""
    for i in range(0, len(srcMacBytes), 2):
        srcMac+=str(srcMacBytes[i:i + 2],'UTF-8')
        srcMac+=":"
    srcMac=srcMac[:-1]
    deviceMac=""
    if srcMac in Mac2Label.keys():
        deviceMac=srcMac
    elif dstMac in Mac2Label.keys():
        deviceMac=dstMac
    return deviceMac

def genData():
    data=[]
    label=[]
    paths = [r'../../../USTC-TK2016/3_ProcessedSession/TrimedSession/Train', r'../../../USTC-TK2016/3_ProcessedSession/TrimedSession/Test']
    for p in paths:
        for d in os.listdir(p):
            for f in os.listdir(os.path.join(p, d)):#f文件名
                file = os.path.join(p, d, f)#一个pcap的完整文件路径
                Sessionmatrix,deviceMac=getMatrixfrom_pcap(file, EmbeddingSize)
                if deviceMac=="":
                    continue
                data.append(Sessionmatrix)
                label.append(Mac2Label[deviceMac])
    #所有文件夹下的所有pcap都解析完了
    data = np.array(data)
    label = np.array(label)
    data = data.reshape((-1, SeqLen*EmbeddingSize))#一行是一个embedding
    label = label.reshape(-1, 1)

    print("data's shape=", data.shape)
    print("label's shape=", label.shape)
    np.savetxt(DataFilePath, data, delimiter=',')
    np.savetxt(LabelFilePath, label, delimiter=',')





class TestSklearn(unittest.TestCase):
    def test_utils(self):
        genData()
    def test_matrix(self):
        getMatrixfrom_pcap(r"D:\Documents\shj\胡伟业我的\iie\PytorchProject\USTC-TK2016\3_ProcessedSession\TrimedSession\Train\16-09-23-ALL\16-09-23.pcap.TCP_13-107-3-128_443_192-168-1-239_35261.pcap",8)