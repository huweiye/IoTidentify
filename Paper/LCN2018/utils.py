import copy

import numpy as np
from sklearn import preprocessing
from flowcontainer.extractor import extract
import os
import unittest

# 在进行自定义数据集的标签设置时，不能从1开始，而是从0开
MACAddress2Label = {
    "44:65:0d:56:cc:d3": 0,  # Amazon Echo
    "d0:52:a8:00:67:5e": 0,  # Smart Things
    "18:b7:9e:02:20:44": 1,  # Tirby Speaker
    "e0:76:d0:33:bb:85": 1,  # PIX-START Photo-frame
    "70:5a:0f:e4:9b:c0": 1,  # HP Printer
    "70:ee:50:18:34:43": 2,  # Netatmo Welcome
    "00:24:e4:11:18:a8": 2,  # Withings Smart Baby Monitor
    "00:16:6c:ab:6b:88": 2,  # Samsung SmartCam
    "f4:f2:6d:93:51:f1": 2,  # TP-Link Day Night Cloud camera
    "30:8c:fb:2f:e4:b2": 2,  # Dropcam
    "00:62:6e:51:27:2e": 2,  # Insteon Camera
    "ec:1a:59:79:f4:89": 3,  # Belkin Wemo switch
    "50:c7:bf:00:56:39": 3,  # TP-Link Smart plug
    "74:c6:3b:29:d7:1d": 3,  # iHome
    "ec:1a:59:83:28:11": 3,  # Belkin wemo motion sensor
}
GateWayMacAddress = "14:cc:20:51:33:ea"
SEQLEN = 6  # 一个序列里单词的个数
FEATURENUM = 6  # 一个embedding6维


class packet:
    def __init__(self, time, len, iscontrol):
        self.time = time  # 当前数据包的epoch time
        self.len = len  # 当前数据包的包长
        self.iscontrol = iscontrol  #


Device2Stream = dict()

UserPackType = {"http", "tls", "ssl", "tcp", "udp"}
ControlPackType = {"icmp", "arp", "dns", "ntp"}


def getFiles(dir, suffix):  # 查找根目录，文件后缀
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename)  # =>文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename))  # =>把一串字符串组合成路径
    return res


def load_dataset():
    extensions = ["eth.src", "eth.dst", "frame.len", "icmp", "arp", "dns", "ntp"]  # 根据MAC地址和载荷长度的正负号来确定当前数据包的label
    file_dir = r"../../../../../../DataSet/DataSet/IoT identification/TMC2018/"
    for file in getFiles(file_dir, '.pcap'):
        # 修改了flowcontainer的源码：注释了把packet组合成flow的步骤，使返回result是一个list，每个元素代表一个包packet（没了这个）
        packetlist = extract(infile=file, filter="", extension=extensions)
        for p in packetlist:
            extension_list = p[10]  # flowcontainer/reader.py:10) extension(s)
            device_mac = ""
            if extension_list[0] == GateWayMacAddress:
                device_mac = extension_list[1]
            else:
                device_mac = extension_list[0]
            if device_mac not in MACAddress2Label:
                continue  # 当前IoT设备没有被论文使用
            is_control = False
            if extension_list[3] != "" or extension_list[4] != "" or extension_list[5] != "" or extension_list[6] != "":
                is_control = True
            if device_mac not in Device2Stream:
                Device2Stream[device_mac] = []
            Device2Stream[device_mac].append(packet(float(p[3]), float(extension_list[2]), is_control))

    time_window = 5 * 60  # 论文里的T，时间窗口是5分钟
    data = []
    seq = []
    label = []
    for device, pack_list in Device2Stream.items():
        # 遍历每个设备
        start_index = 0
        while start_index < len(pack_list):
            end_time = pack_list[start_index].time + time_window
            i = start_index
            # 6个最具鉴别力的特征：用户数据包数量，用户数据包长度平均值，用户数据包长度峰值，控制数据包数量，控制数据包平均值，控制数据包峰值
            num_user = 0
            avelen_user = 0.0
            maxlen_user = 0
            num_control = 0
            avelen_control = 0.0
            maxlen_control = 0
            while i < len(pack_list) and pack_list[i].time <= end_time:
                pack = pack_list[i]
                if pack.iscontrol:
                    num_control += 1
                    avelen_control += pack.len
                    maxlen_control = max(pack.len, maxlen_control)
                else:
                    num_user += 1
                    avelen_user += pack.len
                    maxlen_user = max(pack.len, maxlen_user)
                i += 1
            if num_control != 0:
                avelen_control = (avelen_control / num_control)
            if num_user != 0:
                avelen_user = (avelen_user / num_user)
            start_index = i  # 下一个窗口的起始位置
            seq.append([num_user, avelen_user, maxlen_user, num_control, avelen_control, maxlen_control])
            if len(seq) == SEQLEN:  # 所以说如果当前句子不足一个序列的长度，直接抛弃掉
                data.append(copy.deepcopy(seq))  # 一个句子
                label.append(MACAddress2Label[device])  # 该句子的标签
                seq.clear()
    data = np.array(data)
    label = np.array(label)
    data = data.reshape(-1, FEATURENUM)
    label = label.reshape(-1, 1)

    print("data's shape=", data.shape)
    print("label's shape=", label.shape)
    np.savetxt(r'../../data/lcn2018_data.csv', data, delimiter=',')
    np.savetxt(r'../../data/lcn2018_label.csv', label, delimiter=',')


class TestSklearn(unittest.TestCase):
    def test_utils(self):
        load_dataset()
