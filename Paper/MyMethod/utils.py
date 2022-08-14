#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
from flowcontainer.extractor import extract
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
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
Mac2DeviceName = {
    "d0:52:a8:00:67:5e": "Smart Things",
    "44:65:0d:56:cc:d3": "Amazon Echo ",
    "70:ee:50:18:34:43": "Netatmo Welcome",
    "f4:f2:6d:93:51:f1": "TP-Link Day Night Cloud camera",
    "00:16:6c:ab:6b:88": "Samsung SmartCam",
    "30:8c:fb:2f:e4:b2": "Dropcam",
    "00:62:6e:51:27:2e": "Insteon Camera",
    "00:24:e4:11:18:a8": "Withings Smart Baby Monitor",
    "ec:1a:59:79:f4:89": "Belkin Wemo switch",
    "50:c7:bf:00:56:39": "TP-Link Smart plug",
    "74:c6:3b:29:d7:1d": "iHome",
    "ec:1a:59:83:28:11": "Belkin wemo motion sensor",
    "18:b4:30:25:be:e4": "NEST Protect smoke alarm",
    "70:ee:50:03:b8:ac": "Netatmo weather station",
    "00:24:e4:1b:6f:96": "Withings Smart scale",
    #"74:6a:89:00:2e:25": "Blipcare Blood Pressure meter",
    "00:24:e4:20:28:c6": "Withings Aura smart sleep sensor",
    "d0:73:d5:01:83:08": "Light Bulbs LiFX Smart Bulb",
    "18:b7:9e:02:20:44": "Triby Speaker",
    "e0:76:d0:33:bb:85": "PIX-STAR Photo-frame",
    "70:5a:0f:e4:9b:c0": "HP Printer",
    "08:21:ef:3b:fc:e3": "Samsung Galaxy Tab",
}
LanMac = "14:cc:20:51:33:ea"
SeqLen=10 #每条流只提取前15个数据包的特征?
EmbeddingSize=30#自己算出来的
LabelNum=len(Mac2Label)
Mac2PacketFeatureList = {}  # 每个mac一个列表，列表长度是15的倍数，元素是packetFeature对象
Mac2PktNums={}#mac:包数目

def getFiles(dir, suffix):  # 查找根目录，文件后缀
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename)  # =>文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename))  # =>把一串字符串组合成路径
    return res


class packetFeature:
    def __init__(self,_direction,_deltaTime,_tos,_ttl,_udpOrtcp,_port,_win,_frameLen,deviceMac):
        self.dir=_direction #方向，0入站，1出站
        self.deltaTime=_deltaTime   #距离前一个数据包的时间间隔，frame.time_delta，浮点数
        self.tos=_tos   #区分服务，ip.dsfield，one-hot
        self.ttl=_ttl#跳数,ip.ttl，整数
        self.ipdf=None#设置是否分片，3位，one-hot,ip.flags的前3位~~~~~~~~~~暂时不用
        self.udpORtcp=_udpOrtcp#传输层是udp还是tcp,默认提取
        self.port=_port#服务器端端口号，因为是离散的值所以要embedding
        self.win=_win#自己可接收的窗口大小，udp时为0,tcp.window_size_value
        self.opt=None#tcp.options，最长40个字节,但只取12个字节~~~~~~~~~~~~~~~~~~~~~~~~暂时不用
        self.appProto=None#应用层协议，one-hot表示~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~暂时不用
        self.frameLen=_frameLen#帧长
        self.isEnc=None#数据包是否加密，包括ESP、TLS，私有协议是否加密通过计算字节熵得到~~~~~~~~~~~~~~~暂时不用
        self.deviceMac=deviceMac#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~先试着造一手假

def packet2embedding(pkt:packetFeature)->list:
    '''
    将一个packetFeature对象生成一个特征向量
    :return:
    '''
    embedding=[]

    embedding.append(float(pkt.dir))#int

    embedding.append(float(pkt.deltaTime))#str

    tos_oneHot=bin(int(pkt.tos,16))[2:].rjust(8,'0')#str
    for element in tos_oneHot:
        embedding.append(float(element))

    embedding.append(float(pkt.ttl))#int

    embedding.append(float(pkt.udpORtcp))#int

    server_port_oneHot=bin(pkt.port)[2:].rjust(16,'0') #服务器端端口号是离散值，因为要转换成16位的one-hot
    for element in server_port_oneHot:
        embedding.append(float(element))

    embedding.append(float(pkt.win))#int

    # tcpOptions=bin(int(pkt.opt,16))[2:]
    # for element in tcpOptions:
    #     embedding.append(float(element))

    embedding.append(float(pkt.frameLen))#int

    return embedding



def getDir(srcMacList,pktIndex):#0或1
    for srcMac in srcMacList:
        if srcMac[1]==pktIndex:#找到目标数据包
            if srcMac[0]==LanMac:
                return 0#入站
            else:
                return 1#出站
    return 0

def genDeltaTime(timeDeltaList,pktIndex):#一个浮点数，一般不超过1
    for timeDelta in timeDeltaList:
        if timeDelta[1]==pktIndex:#找到目标数据包
            return timeDelta[0]
    return 0.0

def getTos(tosList,pktIndex):#8位，需要转换成one-hot
    for tos in tosList:
        if tos[1]==pktIndex:
            return tos[0]
    return "0x00"

def getTTL(ttlList,pktIndex):#一个整数
    for ttl in ttlList:
        if ttl[1]==pktIndex:
            return ttl[0]
    return 0

def udpORtcp(ipProto):#0或1
    if ipProto=="tcp":
        return 1
    else:
        return 0
def getWin(winList,pktIndex):#16位，整数值
    for win in winList:
        if win[1]==pktIndex:
            return win[0]
    return 0

def getFrameLen(frameLenList,pktIndex):#帧长，整数
    for frameLen in frameLenList:
        if frameLen[1]==pktIndex:
            return frameLen[0]
    return 0

def getTCPOptions(tcpOptionList,pktIndex):#只取后10个字节，需转换成8*12=96位的one-hot
    for tcpOption in tcpOptionList:
        if tcpOption[1]==pktIndex:
            return "0x"+tcpOption[0][-20:]
    return "0x00000000000000000000"

def printMacPktNum():#打印每个mac的数据包总数
    macLabel=[]
    pktNumList = []
    for mac, deviceName in Mac2DeviceName.items():
        if mac not in Mac2PktNums.keys():
            print("not exist device:", Mac2DeviceName[mac])
            continue
        pktNum=Mac2PktNums[mac]
        print("device {}  packet number= {}".format(Mac2DeviceName[mac], pktNum))
        pktNumList.append(pktNum)
        macLabel.append(deviceName)

    plt.figure(figsize=(20, 10))
    plt.pie(pktNumList, labels=macLabel, autopct='%3.1f%%', pctdistance=0.8)
    plt.title('Iot设备包数目占比图')  # 加标题
    plt.show()

def genMac2PacketFeatureList():
    '''
    :return: data：二维列表，一行是一个流的包长序列；label：一维列表，data对应行的设备类型
    '''
    extensions = ["eth.src", "eth.dst","frame.time_delta","ip.dsfield","ip.ttl","ip.flags","tcp.window_size_value","tcp.options","frame.len"]
    file_dir = r"../../../../../../DataSet/DataSet/IoT identification/TMC2018/TMC2018/test"
    for file in getFiles(file_dir, '.pcap'):
        flowDict = extract(infile=file,  extension=extensions)
        for key in flowDict.keys():#遍历每条流
            flow = flowDict[key]
            srcMacList = flow.extension['eth.src']
            dstMacList = flow.extension['eth.dst']
            timeDeltaList=flow.extension['frame.time_delta']
            tosList=flow.extension['ip.dsfield'] if 'ip.dsfield' in flow.extension.keys() else []
            ttlList=flow.extension['ip.ttl'] if 'ip.ttl' in flow.extension.keys() else []
            ipFlags=flow.extension['ip.flags'] if 'ip.flags' in flow.extension.keys() else []
            winList=flow.extension['tcp.window_size_value'] if 'tcp.window_size_value' in flow.extension.keys() else []
            tcpOptionList=flow.extension['tcp.options'] if 'tcp.options' in flow.extension.keys() else []
            frameLenList =flow.extension['frame.len']
            deviceMac = ""
            if srcMacList[0][0] in Mac2Label.keys():
                deviceMac = srcMacList[0][0]
            elif dstMacList[0][0] in Mac2Label.keys():
                deviceMac = dstMacList[0][0]
            else:#不是已知设备，忽略当前流
                continue
            if deviceMac not in Mac2PktNums.keys():
                Mac2PktNums[deviceMac]=0
            Mac2PktNums[deviceMac]+=len(flow.ip_lengths)
            #后面根据deviceMac确定设备标签
            if deviceMac not in Mac2PacketFeatureList:#当前mac还没有任何元素
                Mac2PacketFeatureList[deviceMac] = []
            for packetIndex in range(0,SeqLen):#取流的前SeqLen个数据包
                dir=getDir(srcMacList,packetIndex)#get 方向
                deltaTime=genDeltaTime(timeDeltaList,packetIndex)
                tos=getTos(tosList,packetIndex)
                ttl=getTTL(ttlList,packetIndex)
                udpTcp=udpORtcp(key[1])
                serverPort=flow.dport#flowcontainer是将较小的端口当作目的端口，这是合理的，因为服务器端固定开放的端口一般比客户端随机绑定的要小
                win=getWin(winList,packetIndex)
                frameLen=getFrameLen(frameLenList,packetIndex)

                Mac2PacketFeatureList[deviceMac].append(packetFeature(_direction=dir,_deltaTime=deltaTime,
                                                                      _tos=tos,_ttl=ttl,_udpOrtcp=udpTcp,
                                                                      _port=serverPort,_win=win,_frameLen=frameLen,deviceMac=deviceMac))

                # print("packet {} feature is:{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{}".format(packetIndex,
                #                                                                                           dir,type(dir),
                #                                                                                           deltaTime,type(deltaTime),
                #                                                                                           tos,type(tos),
                #                                                                                           ttl,type(ttl),
                #                                                                                           udpTcp,type(udpTcp),
                #                                                                                           serverPort,type(serverPort),
                #                                                                                           win,type(win),
                #                                                                                           frameLen,type(frameLen)))
            #一条流提取前SeqLen个数据包
        #当前文件的流都生成完了
    # 所有文件都遍历完了



def genData():
    data=[]
    label=[]
    for deviceMac,pktFeatureList in Mac2PacketFeatureList.items():
        i=0
        for pktFeature in pktFeatureList:
            embedding=packet2embedding(pktFeature)
            data.append(embedding)
            if i%SeqLen==0:
                label.append(Mac2Label[deviceMac])
            i+=1
    data = np.array(data)
    label = np.array(label)
    print("TMC2018 data's shape=", data.shape)
    print("TMC2018 label's shape=", label.shape)
    np.savetxt(r'../../data/TMC2018_data.csv', data, delimiter=',')
    np.savetxt(r'../../data/TMC2018_label.csv', label, delimiter=',')

    for deviceMac,pktFeatureList in Mac2PacketFeatureList.items():
        print("device {} sample count= {},which proportion={}".format(Mac2DeviceName[deviceMac],len(pktFeatureList),len(pktFeatureList)/data.shape[0]))

class TestSklearn(unittest.TestCase):
    def test_utils(self):
        genMac2PacketFeatureList()
        printMacPktNum()
        genData()

    def test_others(self):
        mat1=[
            [
            [1,2,3],[4,5,6]
               ],
            [
                [7,8,9],[10,11,12]
            ]
        ]
        mat1=np.array(mat1)
        print("mat1 shape={}".format(mat1.shape))
        mat1=mat1.reshape(2,-1)
        print(mat1)
        print("mat1 reshape shape={}".format(mat1.shape))

    def test_label(self):#确定一下到底有哪些设备
        hasDevice=set( )
        extensions = ["eth.src", "eth.dst"]
        file_dir = r"../../../../../../DataSet/DataSet/IoT identification/TMC2018/TMC2018/test"
        for file in getFiles(file_dir, '.pcap'):
            flowDict = extract(infile=file, extension=extensions)
            for key in flowDict.keys():  # 遍历每条流
                flow = flowDict[key]
                srcMacList = flow.extension['eth.src']
                dstMacList = flow.extension['eth.dst']
                deviceMac = ""
                if srcMacList[0][0] in Mac2Label.keys():
                    deviceMac = srcMacList[0][0]
                elif dstMacList[0][0] in Mac2Label.keys():
                    deviceMac = dstMacList[0][0]
                else:  # 不是已知设备，忽略当前流
                    continue
                hasDevice.add(deviceMac)
        for deviceMac in Mac2Label.keys():
            if deviceMac not in hasDevice:
                print(deviceMac+"NOT EXIST!!!")















