#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# 分类物联网设备和非物联网设备
import numpy as np
from SHJ import utils
from flowcontainer.extractor import extract
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题

Mac2Label = {
    "d0:52:a8:00:67:5e": 0,  # Smart Things，网关设备
    "44:65:0d:56:cc:d3": 0,  # Amazon Echo，智能音箱
    "70:ee:50:18:34:43": 0,  # Netatmo Welcome，智能摄像头
    "f4:f2:6d:93:51:f1": 0,  # TP-Link Day Night Cloud camera，智能摄像头
    "00:16:6c:ab:6b:88": 0,  # Samsung SmartCam，智能摄像头
    "30:8c:fb:2f:e4:b2": 0,  # Dropcam，智能摄像头
    "00:62:6e:51:27:2e": 0,  # Insteon Camera，智能摄像头
    "00:24:e4:11:18:a8": 0,  # Withings Smart Baby Monitor， 传感器
    "ec:1a:59:79:f4:89": 0,  # Belkin Wemo switch，智能插座
    "50:c7:bf:00:56:39": 0,  # TP-Link Smart plug，智能插座
    "74:c6:3b:29:d7:1d": 0,  # iHome Powerplug，智能插座
    "ec:1a:59:83:28:11": 0,  # Belkin wemo motion sensor，传感器
    "18:b4:30:25:be:e4": 0,  # NEST Protect smoke alarm，传感器
    "70:ee:50:03:b8:ac": 0,  # Netatmo weather station，传感器
    "00:24:e4:1b:6f:96": 0,  # Withings Smart scale，传感器
    "74:6a:89:00:2e:25": 0,  # Blipcare Blood Pressure meter，传感器
    "00:24:e4:20:28:c6": 0,  # Withings Aura smart sleep sensor，传感器
    "d0:73:d5:01:83:08": 0,  # Light Bulbs LiFX Smart Bulb，智能电灯
    "18:b7:9e:02:20:44": 0,  # Triby Speaker，语音助手
    "e0:76:d0:33:bb:85": 0,  # PIX-STAR Photo-frame，相框
    "70:5a:0f:e4:9b:c0": 0,  # HP Printer，打印机
    "30:8c:fb:b6:ea:45": 0,  #Nest Dropcam，智能摄像头，Nest（谷歌子公司）
    "08:21:ef:3b:fc:e3": 1,  # Samsung Galaxy Tab，平板电脑，三星
    "40:f3:08:ff:1e:da": 1,  # Android Phone，
    "74:2f:68:81:69:42": 1,  # Laptop，
    "ac:bc:32:d4:6f:2f": 1,  # MacBook，
    "b4:ce:f6:a7:a3:c2": 1,  # Android Phone，
    "d0:a6:37:df:a1:e1": 1,  # IPhone	，
    "f4:5c:89:93:cc:85": 1,  # MacBook/Iphone，
}
'''定义常量'''
LanMac = "14:cc:20:51:33:ea"
FileDir = r"../../../../../DataSet/DataSet/IoT identification/TMC2018/TMC2018/"
DataPath=r"data/TMC2018 Packet Length Sequence Data.csv"
LabelPath= r'data/TMC2018 Packet Length Sequence Label.csv'
Mac2FlowList = dict()  # <macaddress,该设备下所有的流的包长序列>
FeatureNum = 8
''''''


def feature_trace(trace: list):
    from scipy.stats import skew
    feature = [0.0] * FeatureNum
    if len(trace) == 0:
        return feature
    feature[0] = np.min(trace)
    feature[1] = np.max(trace)
    feature[2] = np.mean(trace)
    feature[3] = np.median(np.absolute(trace - np.mean(trace)))
    feature[4] = np.std(trace)
    feature[5] = np.var(trace)
    feature[6] = skew(trace)
    feature[7] = len(trace)
    return feature

def feature_extract(pkt_length_sequence):
    '''
    :param pkt_length_sequence: 一条流的载荷序列
    :return:
    '''
    trace = []
    pkt_length_sequence = np.array(pkt_length_sequence)
    pkt_length_sequence = pkt_length_sequence.reshape((-1))
    for i in range(pkt_length_sequence.shape[0]):
        trace.append(pkt_length_sequence[i])
    feature = feature_trace(trace)
    return feature

def genPayloadLens(file_dir:str):
    '''
    生成设备mac地址到包长序列的map
    '''
    extensions = ["eth.src", "eth.dst"]
    UsePcapNum=10
    for file in utils.getFiles(file_dir, '.pcap'):
        UsePcapNum-=1
        if UsePcapNum==0:
            break
        flowDict = extract(infile=file, filter="(tcp or udp)", extension=extensions)
        for key in flowDict:
            flow = flowDict[key]
            payloadLensList = flow.payload_lengths
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
            Mac2FlowList[deviceMac].append(payloadLensList)

def GenData(file_dir:str):
    '''
    生成数据csv文件：data：（样本数*包长序列特征数）一行是一个流的包长序列，label：data对应行的设备类型
    :param file_dir: 包含pcap文件的路径
    '''
    genPayloadLens(file_dir)
    data = []
    label = []
    for deviceMac, pktLensList in Mac2FlowList.items():
        for pktLens in pktLensList:
            data.append(feature_extract(pktLens))
            label.append(Mac2Label[deviceMac])
    data = np.array(data)
    label = np.array(label)
    print("TMC2018 Packet Length Sequence Data shape=", data.shape)
    print("TMC2018 Packet Length Sequence Label shape=", label.shape)
    np.savetxt(DataPath, data, delimiter=',')
    np.savetxt(LabelPath, label, delimiter=',')
    for deviceMac,pktLensList in Mac2FlowList.items():
        print("device {} sample count= {},which proportion={:.1%}".format(deviceMac,len(pktLensList),len(pktLensList)/data.shape[0]))

def ClassifyIoTAndNonIoT(data: np.ndarray, label: np.ndarray):
    data = data.reshape(-1, FeatureNum)
    label = label.reshape(label.size)
    #归一化
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(data)
    data = scaler.transform(data)
    #划分数据集
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)
    #随机森林
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(random_state=1)
    rfc.fit(X_train, y_train)
    score_r = rfc.score(X_test, y_test)
    print("Random Forest score:{}".format(score_r))#在测试集上的平均准确度
    y_predict = rfc.predict(X_test)
    matrix = confusion_matrix(y_test, y_predict, labels=[0, 1], normalize='true')
    print(matrix)#测试集上的混淆矩阵
    from sklearn.metrics import precision_score, recall_score, f1_score
    print(precision_score(y_test, y_predict))
    print(recall_score(y_test, y_predict, average='micro'))
    print(f1_score(y_test, y_predict, average='weighted'))
    #展示特征重要性
    f, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(len(rfc.feature_importances_)), rfc.feature_importances_)
    ax.set_title("Feature Importances")
    plt.show()