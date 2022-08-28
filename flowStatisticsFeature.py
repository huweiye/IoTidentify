#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pickle
import random

import numpy as np
from flowcontainer.extractor import extract
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

'''
Mac2Label={
"d0:52:a8:00:67:5e":0 ,#Smart Things
"44:65:0d:56:cc:d3":1 ,#Amazon Echo
"70:ee:50:18:34:43":2 ,#Netatmo Welcome
"f4:f2:6d:93:51:f1":3 ,#TP-Link Day Night Cloud camera
"00:16:6c:ab:6b:88":4 ,#Samsung SmartCam
"30:8c:fb:2f:e4:b2":5 ,#Dropcam
"00:62:6e:51:27:2e":6 ,#Insteon Camera
"00:24:e4:11:18:a8":7 ,#Withings Smart Baby Monitor
"ec:1a:59:79:f4:89":8 ,#Belkin Wemo switch
"50:c7:bf:00:56:39":9 ,#TP-Link Smart plug
"74:c6:3b:29:d7:1d":10 ,#iHome
"ec:1a:59:83:28:11":11 ,#Belkin wemo motion sensor
"18:b4:30:25:be:e4":12 ,#NEST Protect smoke alarm
"70:ee:50:03:b8:ac":13 ,#Netatmo weather station
"00:24:e4:1b:6f:96":14 ,#Withings Smart scale
"74:6a:89:00:2e:25":15 ,#Blipcare Blood Pressure meter
"00:24:e4:20:28:c6":16 ,#Withings Aura smart sleep sensor
"d0:73:d5:01:83:08":17 ,#Light Bulbs LiFX Smart Bulb
"18:b7:9e:02:20:44":18 ,#Triby Speaker
"e0:76:d0:33:bb:85":19 ,#PIX-STAR Photo-frame
"70:5a:0f:e4:9b:c0":20 ,#HP Printer
"08:21:ef:3b:fc:e3":21 ,#Samsung Galaxy Tab
"40:f3:08:ff:1e:da":22 ,#Non-Iot
"74:2f:68:81:69:42":22 ,#Non-Iot
"ac:bc:32:d4:6f:2f":22 ,#Non-Iot
"b4:ce:f6:a7:a3:c2":22 ,#Non-Iot
"d0:a6:37:df:a1:e1":22 ,#Non-Iot
"f4:5c:89:93:cc:85":22 ,#Non-Iot
}
'''
Mac2Label = {
    "d0:52:a8:00:67:5e": 0,  # Smart Things
    "44:65:0d:56:cc:d3": 0,  # Amazon Echo
    "70:ee:50:18:34:43": 0,  # Netatmo Welcome
    "f4:f2:6d:93:51:f1": 0,  # TP-Link Day Night Cloud camera
    "00:16:6c:ab:6b:88": 0,  # Samsung SmartCam
    "30:8c:fb:2f:e4:b2": 0,  # Dropcam
    "00:62:6e:51:27:2e": 0,  # Insteon Camera
    "00:24:e4:11:18:a8": 0,  # Withings Smart Baby Monitor
    "ec:1a:59:79:f4:89": 0,  # Belkin Wemo switch
    "50:c7:bf:00:56:39": 0,  # TP-Link Smart plug
    "74:c6:3b:29:d7:1d": 0,  # iHome
    "ec:1a:59:83:28:11": 0,  # Belkin wemo motion sensor
    "18:b4:30:25:be:e4": 0,  # NEST Protect smoke alarm
    "70:ee:50:03:b8:ac": 0,  # Netatmo weather station
    "00:24:e4:1b:6f:96": 0,  # Withings Smart scale
    "74:6a:89:00:2e:25": 0,  # Blipcare Blood Pressure meter
    "00:24:e4:20:28:c6": 0,  # Withings Aura smart sleep sensor
    "d0:73:d5:01:83:08": 0,  # Light Bulbs LiFX Smart Bulb
    "18:b7:9e:02:20:44": 0,  # Triby Speaker
    "e0:76:d0:33:bb:85": 0,  # PIX-STAR Photo-frame
    "70:5a:0f:e4:9b:c0": 0,  # HP Printer
    "08:21:ef:3b:fc:e3": 0,  # Samsung Galaxy Tab
    "40:f3:08:ff:1e:da": 1,  # Non-Iot
    "74:2f:68:81:69:42": 1,  # Non-Iot
    "ac:bc:32:d4:6f:2f": 1,  # Non-Iot
    "b4:ce:f6:a7:a3:c2": 1,  # Non-Iot
    "d0:a6:37:df:a1:e1": 1,  # Non-Iot
    "f4:5c:89:93:cc:85": 1,  # Non-Iot
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
    "74:6a:89:00:2e:25": "Blipcare Blood Pressure meter",
    "00:24:e4:20:28:c6": "Withings Aura smart sleep sensor",
    "d0:73:d5:01:83:08": "Light Bulbs LiFX Smart Bulb",
    "18:b7:9e:02:20:44": "Triby Speaker",
    "e0:76:d0:33:bb:85": "PIX-STAR Photo-frame",
    "70:5a:0f:e4:9b:c0": "HP Printer",
    "08:21:ef:3b:fc:e3": "Samsung Galaxy Tab",
    "40:f3:08:ff:1e:da": "Non-Iot",
    "74:2f:68:81:69:42": "Non-Iot",
    "ac:bc:32:d4:6f:2f": "Non-Iot",
    "b4:ce:f6:a7:a3:c2": "Non-Iot",
    "d0:a6:37:df:a1:e1": "Non-Iot",
    "f4:5c:89:93:cc:85": "Non-Iot"
}

IotNum = 22

LanMac = "14:cc:20:51:33:ea"

Mac2FlowList = dict()  # <macaddress,该设备下所有的流 每个流一个包长序列>

FeatureNum = 7


def getFiles(dir, suffix):  # 查找根目录，文件后缀
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename)  # =>文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename))  # =>把一串字符串组合成路径
    return res


def loadDataset():
    '''
    :return: data,一行是一个流的包长序列
    label：data对应行的设备类型
    '''
    extensions = ["eth.src", "eth.dst"]
    file_dir = r"../../../../DataSet/DataSet/IoT identification/TMC2018/TMC2018/test"
    for file in getFiles(file_dir, '.pcap'):
        flowDict = extract(infile=file, filter="(tcp or udp)", extension=extensions)
        for key in flowDict:
            flow = flowDict[key]
            payloadLenList = flow.ip_lengths
            srcMacList = flow.extension['eth.src']
            dstMacList = flow.extension['eth.dst']
            deviceMac = ""
            if srcMacList[0][0] == LanMac:  # 是入站数据包
                deviceMac = dstMacList[0][0]
            elif dstMacList[0][0] == LanMac:  # 是出站数据包
                deviceMac = srcMacList[0][0]
            if deviceMac not in Mac2Label.keys():  # 不是已知设备
                continue
            if deviceMac not in Mac2FlowList:
                Mac2FlowList[deviceMac] = []
            Mac2FlowList[deviceMac].append(payloadLenList)

    # 因为Non-IoT流少，IoT流多，所以对IoT流进行欠采样，使其总数目与Non-IoT保持一致
    flowNumIot = 0
    flowNumNonIot = 0
    for deviceMac, pktLensList in Mac2FlowList.items():
        if Mac2Label[deviceMac] == 0:
            flowNumIot += len(pktLensList)
        else:
            flowNumNonIot += len(pktLensList)
    dec = (flowNumNonIot / flowNumIot)  # Non-Iot设备的总流数目占Iot设备总流数目的几分之几，后面对每个Iot设备，以dec的概率选择样本
    data = []
    label = []
    for deviceMac, pktLensList in Mac2FlowList.items():
        for pktLens in pktLensList:
            # 胡伟业：for pktLens in random.sample(pktLensList,int(dec*len(pktLensList))):# 对IoT设备的流量进行欠采样
            from Routers import utils
            data.append(feature_extract(pktLens))
            label.append(Mac2Label[deviceMac])
    return np.array(data), np.array(label)


def feature_trace(trace: list):
    feature = [0.0] * FeatureNum
    if len(trace) == 0:
        return feature
    feature[0] = np.min(trace)
    feature[1] = np.max(trace)
    feature[2] = np.mean(trace)
    feature[3] = np.median(np.absolute(trace - np.mean(trace)))
    feature[4] = np.std(trace)
    feature[5] = np.var(trace)
    feature[6] = len(trace)
    return feature

def feature_extract(pkt_length_sequence):  # 根据包长序列返回特征向量
    trace = []
    pkt_length_sequence = np.array(pkt_length_sequence)
    pkt_length_sequence = pkt_length_sequence.reshape((-1))
    for i in range(pkt_length_sequence.shape[0]):
        trace.append(pkt_length_sequence[i])
    feature = feature_trace(trace)
    return feature

def classifyIot(data: np.ndarray, label: np.ndarray):
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(data)
    data = scaler.transform(data)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(random_state=1)
    rfc.fit(X_train, y_train)
    score_r = rfc.score(X_test, y_test)
    print("Random Forest score:{}".format(score_r))
    y_predict = rfc.predict(X_test)
    maxtrix = confusion_matrix(y_test, y_predict, labels=[0, 1], normalize='true')
    print(maxtrix)
    print(rfc.feature_importances_)  # 输出这个就可以得到特征重要性，但是只有数值，不具有可读性
    disp = ConfusionMatrixDisplay(confusion_matrix=maxtrix, display_labels=['Iot', 'Non-Iot'])
    disp.plot(cmap='Greens')
    plt.show()

    # save model
    f = open('saved_model/rfc.pickle', 'wb+')
    pickle.dump(rfc, f)
    f.close()

    '''对随机森林模型做十倍交叉验证
    rfc_l = []
    for i in range(10):
        rfc = RandomForestClassifier(n_estimators=25)
        rfc_s = cross_val_score(rfc, data, label, cv=10).mean()
        rfc_l.append(rfc_s)
    plt.plot(range(1, 11), rfc_l, label="Random Forest")
    plt.legend(title='10-fold cross-validation')
    plt.show()
    '''


def myDrawing():  # 画出不同IoT设备包数的分布
    # 图1：全部设备的流数目的分布饼状图（实际上因为文字重叠不好看，有部分没显示）
    def pieChart():
        flowNumList = []  # 22个物联网和非物联网设备，共23个元素
        macLabel = []
        NonIotNum = 0
        for mac, deviceName in Mac2DeviceName.items():
            if mac not in Mac2FlowList.keys():
                print("not exist device:", mac)
                continue
            print("the flow number of {} is {}".format(Mac2DeviceName[mac], len(Mac2FlowList[mac])))
            if Mac2Label[mac] == 0:
                flowNumList.append(len(Mac2FlowList[mac]))
                macLabel.append(deviceName)
            else:
                NonIotNum += len(Mac2FlowList[mac])
        flowNumList.append(NonIotNum)
        macLabel.append("Non-Iot")

        plt.figure(figsize=(20, 10))
        plt.pie(flowNumList, labels=macLabel, autopct='%3.1f%%', pctdistance=0.8)
        plt.title('Iot设备流数目占比图')  # 加标题
        plt.show()

    # 图2：做5个IoT设备和Non-IoT设备的包数目占比饼状图，为了说明实际情况下非IoT流量远远大于IoT
    def histoGram():
        sampleMac2Name = {
            "d0:73:d5:01:83:08": "Light Bulbs LiFX Smart Bulb",  # 智能灯泡，能源设备
            "70:5a:0f:e4:9b:c0": "HP Printer",  # 打印机，办公设备
            "44:65:0d:56:cc:d3": "Amazon Echo",  # 控制设备
            "70:ee:50:03:b8:ac": "Netatmo weather station",  # 传感器设备
            "f4:f2:6d:93:51:f1": "TP-Link Day Night Cloud camera",  # 视频设备
            "40:f3:08:ff:1e:da": "Non-Iot",
            "74:2f:68:81:69:42": "Non-Iot",
            "ac:bc:32:d4:6f:2f": "Non-Iot",
            "b4:ce:f6:a7:a3:c2": "Non-Iot",
            "d0:a6:37:df:a1:e1": "Non-Iot",
            "f4:5c:89:93:cc:85": "Non-Iot",
        }
        flowNumList = []
        macLabel = []
        NonIotNum = 0
        for mac, deviceName in sampleMac2Name.items():
            if Mac2Label[mac] == 0:
                flowNumList.append((len(Mac2FlowList[mac]) if mac in Mac2FlowList.keys() else 0))
                macLabel.append(deviceName)
            else:
                NonIotNum += (len(Mac2FlowList[mac]) if mac in Mac2FlowList.keys() else 0)
        flowNumList.append(NonIotNum)
        macLabel.append("Non-Iot")

        plt.pie(flowNumList, labels=macLabel, autopct='%3.1f%%')
        plt.title('Iot设备流数目占比图')  # 加标题
        plt.show()
        # 做包长折线图

    # 图3：做5个IoT设备和Non-IoT设备的包数目随时间的折线图，为了说明同一时刻下非IoT包数目总是大于IoT

    # 图4：做IoT和Non-IoT的包长区间频数统计图，为了说明用包长可以区分IoT和非IoT

    #图5：做IoT和非IoT流持续时间统计图

    #图6：做IoT设备的每条流里包的个数的分布图


if __name__ == '__main__':
    data, label = loadDataset()
    myDrawing()  # 对样本占比做饼状图
    classifyIot(data, label)
