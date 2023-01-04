#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#流量载荷转数据
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import binascii
import os

#小米设备的mac和标签
Mac2Label = {
    "28:6c:07:87:54:b0":0,#mitu_story
    "78:11:dc:cf:c8:f1":1,#xiaobai_camera
    "78:11:dc:e1:f0:6b":2,#xiaomi_control_plug
}
SeqLen = 10  #会话截取的数据包数
EmbeddingSize = 800  # 截取数据包的字节数
LabelNum = len(Mac2Label)
DataFilePath = r'data/MIotPayload Data.csv'
LabelFilePath = r'data/MIotPayload Label.csv'

def getSessionSample(filename):
    '''
    从一个会话里获取若干embedding，跳过以太网帧头，从IP层开始截取
    :param filename: pcap文件路径
    :return:如果会话的mac属于Mac2Label，则返回若干embedding组成的列表，否则返回[]
    '''
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)#返回二进制数据的十六进制表示
    # Pcap 文件全局报头的长度为 24 字节
    hexst = hexst[24 * 2:]
    sessionLen=len(hexst)
    res = []
    #因为小米设备数据量不够，所以不同于TMC2018，指定截取会话的前SeqLen个数据包的做法，我这里是使用会话里全部数据包，最后再用SeqLen切分
    deviceMac = getDeviceMac(hexst)  # 获取设备Mac地址
    if deviceMac == "":#不属于Mac2Label的mac，不需要该会话
        return res, deviceMac
    packetHeader = 0  # 指向每个数据包的捕获记录(packet Record)的首字节

    while packetHeader < sessionLen:
        try:
            i = packetHeader  # 现在i指向packet header的首字节
            i += (16 * 2)  # 跳过Packet Header（固定16字节），现在i指向packet的首字节
            i += (14 * 2)  # 跳过以太网帧格式（），不要Mac地址了，现在i指向ip层的首字节
            packetLen = int(hexst[packetHeader + 8 * 2:(packetHeader + 8 * 2) + 4 * 2], 16)  # 数据帧的实际长度，以字节为单位
            packetData = [int(hexst[i:i + 2], 16) for i in
                          range(i, i + min((packetLen-14)*2, EmbeddingSize * 2), 2)]  # 截取当前数据包指定长度
            # 对当前数据包隐藏ip地址
            for indexIP in range(12, min(12 + 8, len(packetData))):
                packetData[indexIP] = 0
            if len(packetData) < EmbeddingSize:#当前数据包不够大
                for i in range(0, EmbeddingSize - len(packetData)):
                    packetData.append(0)  # 当前packet不足embedding的补上若干个0
            res.append(packetData)  # 在当前会话里，添加上当前packet的payload
            packetHeader += (16 * 2 + packetLen * 2)  # 下一个Packet Record记录的起始位置
        except IndexError:
            break  # 发生越界直接break出循环
    return res, deviceMac


def getDeviceMac(packetRecordList):
    '''
    找到十六进制文件里的Iot设备mac，如果是不存在的设备则为""
    :param packetRecordList:
    :return:
    '''
    # 对pcap文件，每个数据包的捕获记录（Packet Record）都由Packet Header + Packet Data组成
    packetData = packetRecordList[16 * 2:]  # Packet Header 长度固定为 16 字节,Packet Header之后才是第一个数据包的Data
    # 通过找第一个数据包的srcMac和dstMac确定Label
    dstMacBytes = packetData[:6 * 2]
    srcMacBytes = packetData[6 * 2:6 * 2 + 6 * 2]#mac地址6个字节
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
    path = r'../../USTC-TK2016/2_Session/AllLayers'
    for d in os.listdir(path):#path下面是以划分会话之前的pcap为名字命名的文件夹
        for f in os.listdir(os.path.join(path, d)):  # pcap文件，每个pcap是一个会话
            file = os.path.join(path, d, f)  # 一个pcap的完整文件路径
            Sessionmatrix, deviceMac = getSessionSample(file)
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
    from collections import Counter
    print("不同类别的样本数：",dict(Counter(label)))
    np.savetxt(DataFilePath, data, delimiter=',')
    np.savetxt(LabelFilePath, label, delimiter=',')