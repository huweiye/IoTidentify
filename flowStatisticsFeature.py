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
import seaborn as sns
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
"40:f3:08:ff:1e:da":21 ,#Non-Iot
"74:2f:68:81:69:42":21 ,#Non-Iot
"ac:bc:32:d4:6f:2f":21 ,#Non-Iot
"b4:ce:f6:a7:a3:c2":21 ,#Non-Iot
"d0:a6:37:df:a1:e1":21 ,#Non-Iot
"f4:5c:89:93:cc:85":21 ,#Non-Iot
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
    "08:21:ef:3b:fc:e3": 1,  # Samsung Galaxy Tab
    "40:f3:08:ff:1e:da": 1,  # Non-Iot
    "74:2f:68:81:69:42": 1,  # Non-Iot
    "ac:bc:32:d4:6f:2f": 1,  # Non-Iot
    "b4:ce:f6:a7:a3:c2": 1,  # Non-Iot
    "d0:a6:37:df:a1:e1": 1,  # Non-Iot
    "f4:5c:89:93:cc:85": 1,  # Non-Iot
}
Mac2DeviceName = {
    "d0:52:a8:00:67:5e": "Samsung Smart Things",#Audio
    "44:65:0d:56:cc:d3": "Amazon Echo",#Audio
    "70:ee:50:18:34:43": "Netatmo Welcome",#Camera
    "f4:f2:6d:93:51:f1": "TP-Link Day Night Cloud camera",#Camera
    "00:16:6c:ab:6b:88": "Samsung SmartCam",#Camera
    "30:8c:fb:2f:e4:b2": "Dropcam",#Camera  ,google子公司nest收购
    "00:62:6e:51:27:2e": "Insteon Camera",#,Camera  ,2022年4月倒闭
    "00:24:e4:11:18:a8": "Withings Smart Baby Monitor",#Camera,  法国公司
    "ec:1a:59:79:f4:89": "Belkin Wemo switch",#Electronic Switch,贝尔金公司
    "50:c7:bf:00:56:39": "TP-Link Smart plug",#Electronic Switch,
    "74:c6:3b:29:d7:1d": "iHome power plug",#Electronic Switch,
    "ec:1a:59:83:28:11": "Belkin wemo motion sensor",#Sensor
    "18:b4:30:25:be:e4": "NEST Protect smoke alarm",#Sensor
    "70:ee:50:03:b8:ac": "Netatmo weather station",#Sensor
    "00:24:e4:1b:6f:96": "Withings Smart scale",#Sensor
    "74:6a:89:00:2e:25": "Blipcare Blood Pressure meter",#Sensor
    "00:24:e4:20:28:c6": "Withings Aura smart sleep sensor",#Sensor
    "d0:73:d5:01:83:08": "Light Bulbs LiFX Smart Bulb",#Electronic Switch
    "18:b7:9e:02:20:44": "Triby Speaker",#Audio, Invoxia
    "e0:76:d0:33:bb:85": "PIX-STAR Photo-frame",#PIX-STAR,算什么设备，不知道，算其他吧
    "70:5a:0f:e4:9b:c0": "HP Printer",#HP，Electronic Switch，算其他吧
    "08:21:ef:3b:fc:e3": "Non-Iot",#
    "40:f3:08:ff:1e:da": "Non-Iot",
    "74:2f:68:81:69:42": "Non-Iot",
    "ac:bc:32:d4:6f:2f": "Non-Iot",
    "b4:ce:f6:a7:a3:c2": "Non-Iot",
    "d0:a6:37:df:a1:e1": "Non-Iot",
    "f4:5c:89:93:cc:85": "Non-Iot"
}
DeviceMac2Type={
"d0:52:a8:00:67:5e": "Audio",
    "44:65:0d:56:cc:d3": "Audio",
    "70:ee:50:18:34:43": "Camera",
    "f4:f2:6d:93:51:f1": "Camera",
    "00:16:6c:ab:6b:88": "Camera",
    "30:8c:fb:2f:e4:b2": "Camera",
    "00:62:6e:51:27:2e": "Camera",
    "00:24:e4:11:18:a8": "Camera",# 法国公司
    "ec:1a:59:79:f4:89": "Electronic Switch",#,贝尔金公司
    "50:c7:bf:00:56:39": "Electronic Switch",#Electronic Switch,
    "74:c6:3b:29:d7:1d": "Electronic Switch",#Electronic Switch,
    "ec:1a:59:83:28:11": "Sensor",#Sensor
    "18:b4:30:25:be:e4": "Sensor",#Sensor
    "70:ee:50:03:b8:ac": "Sensor",#Sensor
    "00:24:e4:1b:6f:96": "Sensor",#Sensor
    "74:6a:89:00:2e:25": "Sensor",#Sensor
    "00:24:e4:20:28:c6": "Sensor",#Sensor
    "d0:73:d5:01:83:08": "Electronic Switch",#Electronic Switch
    "18:b7:9e:02:20:44": "Audio",#Audio, Invoxia
    "e0:76:d0:33:bb:85": "Others",#PIX-STAR,
    "70:5a:0f:e4:9b:c0": "Others",#HP
}
Vendor2DeviceMac={
    'Samsung':['d0:52:a8:00:67:5e','00:16:6c:ab:6b:88'],#三星
    'Amazon':['44:65:0d:56:cc:d3'],
    'Netatmo':['70:ee:50:18:34:43',"70:ee:50:03:b8:ac"],
    'TP-Link':['f4:f2:6d:93:51:f1','50:c7:bf:00:56:39'],
    'NEST':['30:8c:fb:2f:e4:b2','18:b4:30:25:be:e4'],#Dropcam已被google子公司nest收购，因此放到NEST里
    'Insteon':['00:62:6e:51:27:2e'],#2022年4月倒闭
    'Withings':['00:24:e4:11:18:a8','00:24:e4:1b:6f:96','00:24:e4:20:28:c6'],#法国公司
    'Belkin':['ec:1a:59:79:f4:89','ec:1a:59:83:28:11'],#贝尔金公司
    "iHome":["74:c6:3b:29:d7:1d",],
    'Blipcare':['74:6a:89:00:2e:25'],
    'LIFX':['d0:73:d5:01:83:08'],
    'Invoxia':['18:b7:9e:02:20:44'],
    'PIX-STAR':['e0:76:d0:33:bb:85'],
    'HP':['70:5a:0f:e4:9b:c0']
}

def printIoTDevice():
    import pandas as pd
    data = {"Vendor":[],"Device":[],"Type":[]}
    for vendor,macList in Vendor2DeviceMac.items():
        for mac in macList:
            data['Vendor'].append(vendor)
            data['Device'].append(Mac2DeviceName[mac])
            data['Type'].append(DeviceMac2Type[mac])

    info = pd.DataFrame(data)
    csv_data = info.to_csv('Data.csv')
    print('\nCSV String Values:\n', csv_data)



IotNum = 21

LanMac = "14:cc:20:51:33:ea"

Mac2FlowList = dict()  # <macaddress,该设备下所有的流 每个流一个包长序列>

FeatureNum = 8


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
    :return: data,一行是一个流的包长序列；label：data对应行的设备类型
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

    data = []
    label = []
    for deviceMac, pktLensList in Mac2FlowList.items():
        for pktLens in pktLensList:
            #from Routers.utils import feature_extract
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
    from scipy.stats import skew, kurtosis
    feature[6] = skew(trace)
    feature[7] = len(trace)
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
    from sklearn.metrics import precision_score,recall_score,f1_score
    print(precision_score(y_test, y_predict))
    print(recall_score(y_test, y_predict, average='micro'))
    print(f1_score(y_test, y_predict, average='weighted'))

    import pandas as pd
    features = ['最小值','最大值','平均数','中位数','标准差','方差','偏度','包数目']
    feature_importances = rfc.feature_importances_
    features_df = pd.DataFrame({'Features': features, 'Importance': feature_importances})
    features_df.sort_values('Importance', inplace=True, ascending=False)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题
    sns.barplot(features_df['Features'][:10], features_df['Importance'][:10], )
    plt.xlabel('特征')
    plt.ylabel('特征重要程度')
    # 数据可视化：柱状图
    sns.despine(bottom=True)
    plt.show()

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
    # 图1：做5个IoT设备和Non-IoT设备的一条流中的包数频数分布图
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
    NonIoTpktLenList=[]
    IoTpktLenList=[]
    for deviceMac, pktLensList in Mac2FlowList.items():
        if deviceMac in sampleMac2Name.keys():
            if sampleMac2Name[deviceMac]=='Non-Iot':
                for pktLens in pktLensList:
                    for pktLen in pktLens:
                        NonIoTpktLenList.append(abs(pktLen))
            else:
                for pktLens in pktLensList:
                    for pktLen in pktLens:
                        IoTpktLenList.append(abs(pktLen))
    #drawHist(NonIoTpktLenList,IoTpktLenList)

    NonIoTFlowPktNum=[]
    IoTFlowPktNum=[]
    for deviceMac, pktLensList in Mac2FlowList.items():
        if deviceMac in Mac2DeviceName.keys():
            if Mac2DeviceName[deviceMac]=='Non-Iot':
                for pktLens in pktLensList:
                    NonIoTFlowPktNum.append(len(pktLens))
            else:
                for pktLens in pktLensList:
                    IoTFlowPktNum.append(len(pktLens))
    #drawBar(NonIoTFlowPktNum,IoTFlowPktNum)



def drawBar(data1,data2):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题
    y1 = []
    for i in range(0,40):
        y1.append(data1.count(i))

    y2=[]
    for i in range(0,40):
        y2.append(data2.count(i))

    plt.scatter(range(0,40), y1,label="Non-IoT")
    plt.scatter(range(0,40),y2,c='red',marker='*',label='IoT')
    plt.xlabel("一条流的包数目")
    plt.ylabel("流数目")
    plt.title("流内包数目统计图")
    plt.legend(loc='best')
    plt.show()



def drawHist(dist_data_1,dist_data_2):
    fig, axes = plt.subplots(1, 2)
    # 生成2*2的画布
    plt.subplot(1, 2, 1)

    # 画布中的第一张图
    sns.distplot(dist_data_1, ax=axes[0])
    axes[0].set_title("Non-IoT设备包长分布")
    axes[0].set_xlabel("packet length/bytes")

    plt.subplot(1, 2, 2)
    sns.distplot(dist_data_2, ax=axes[1])
    axes[1].set_title("IoT设备包长分布")
    axes[1].set_xlabel("packet length/bytes")
    plt.show()



if __name__ == '__main__':
    printIoTDevice()

    def test_utils():
        data, label = loadDataset()
        myDrawing()  # 对样本占比做饼状图
        classifyIot(data, label)