import functools

import numpy as np
from sklearn import preprocessing
from flowcontainer.extractor import extract
import os
import unittest
import pandas as pd

'''
可能代码复现错了，应该是先分流，按照流分窗口
'''

#在进行自定义数据集的标签设置时，不能从1开始，而是从0开
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
Device2Stream = dict()#label:packet_embedding的列表
EmbeddingSize=9+16
SeqLen=100#论文中的时间窗口，取100个数据包，相当于seq_len's=100
LabelNum=len(Mac2Label)
Mac2PacketFeatureList = {}  # 每个mac一个列表，列表长度是10的倍数，元素是packetFeature对象
OriginDataFilePath=r'D:\Documents\shj\胡伟业我的\iie\PytorchProject\Pytorch\data/AsiaCCS2020_data.csv'
OriginLabelFilePath=r'D:\Documents\shj\胡伟业我的\iie\PytorchProject\Pytorch\data/AsiaCCS2020_label.csv'
def load_dataset():
    extensions=["eth.src", "eth.dst","frame.len","frame.time_delta","ip","tcp","udp","tls","http","dns"]#目的端口（value.dport）在flowcontainer中默认提取了
    file_dir =  r"../../../../../../DataSet/DataSet/IoT identification/TMC2018/TMC2018/test"
    for file in getFiles(file_dir, '.pcap'):
        flowDict = extract(infile=file, extension=extensions)
        for key in flowDict.keys():  # 遍历每条流
            flow = flowDict[key]
            srcMacList = flow.extension['eth.src']
            dstMacList = flow.extension['eth.dst']
            frameLenList=flow.extension['frame.len']
            timeEpoch=flow.extension['frame.time_delta']
            ipList=flow.extension['ip'] if 'ip' in flow.extension.keys() else []
            tcpList=flow.extension['tcp'] if 'tcp' in flow.extension.keys() else []
            udpList=flow.extension['udp'] if 'udp' in flow.extension.keys() else []
            tlsList=flow.extension['tls'] if 'tls' in flow.extension.keys() else []
            httpList=flow.extension['http'] if 'http' in flow.extension.keys() else []
            dnsList=flow.extension['dns'] if 'dns' in flow.extension.keys() else []
            dport=flow.dport
            dportOneHot=[]
            for s in str(bin(dport)[2:]).rjust(16,'0'):
                dportOneHot.append(float(s))
            deviceMac = ""
            maxPktNum=len(frameLenList)#一条流的包数目
            if srcMacList[0][0] in Mac2Label.keys():
                deviceMac = srcMacList[0][0]
            elif dstMacList[0][0] in Mac2Label.keys():
                deviceMac = dstMacList[0][0]
            else:  # 不是已知设备，忽略当前流
                continue
            if deviceMac not in Mac2PacketFeatureList:#当前mac还没有任何元素
                Mac2PacketFeatureList[deviceMac] = []
            for packetIndex in range(0,maxPktNum):
                pkt=[]
                pkt.extend(dportOneHot)
                pkt.append(float(len(ipList)!=0))
                pkt.append(float(len(tcpList) != 0))
                pkt.append(float(len(udpList) != 0))
                pkt.append(float(len(tlsList) != 0))
                pkt.append(float(len(httpList) != 0))
                pkt.append(float(len(dnsList) != 0))
                pkt.append(float(srcMacList[packetIndex][0]==deviceMac))
                pkt.append(float(frameLenList[packetIndex][0]))
                pkt.append(float(timeEpoch[packetIndex][0]))
                #print(pkt)
                Mac2PacketFeatureList[deviceMac].append(pkt)

def genData():
    data=[]
    label=[]
    for deviceMac,pktFeatureList in Mac2PacketFeatureList.items():
        i=0
        sampleNum=SeqLen*(len(pktFeatureList)//SeqLen)
        for pktIndex in range(sampleNum):
            data.append(pktFeatureList[pktIndex])
            if i%SeqLen==0:
                label.append(Mac2Label[deviceMac])
            i+=1
    data = np.array(data)
    label = np.array(label)
    print("AsiaCCS2020 data's shape=", data.shape)
    print("AsiaCCS2020 label's shape=", label.shape)
    np.savetxt(OriginDataFilePath, data, delimiter=',')
    np.savetxt(OriginLabelFilePath, label, delimiter=',')

    for deviceMac,pktFeatureList in Mac2PacketFeatureList.items():
        print("device {} sample count= {},which proportion={:.1%}".format(Mac2Label[deviceMac],len(pktFeatureList),len(pktFeatureList)/data.shape[0]))




def getFiles(dir, suffix):  # 查找根目录，文件后缀
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename)  # =>文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename))  # =>把一串字符串组合成路径
    return res



class TestLoadData(unittest.TestCase):
    def test_utils(self):
        load_dataset()
        genData()
