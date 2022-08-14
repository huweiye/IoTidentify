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
MACAddress2Label = {
    "88:71:e5:ed:be:c7": 0,  # echo dot
    "f4:f5:d8:db:61:84": 1,  # google home
    "18:bc:5a:19:eb:7d": 2,  # tmall assist
    "78:11:dc:e1:f0:6b": 3,  # xiaomi hub
    "b0:59:47:34:16:ff": 4,  # 360 camera
    "78:11:dc:cf:c8:f1": 5,  # xiaobai camera
    "30:20:10:fb:7c:05": 6,  # tplink plug
    "b4:e6:2d:08:63:0c": 7,  # orvibo plug
    "78:0f:77:1b:00:8c": 8,  # broadlink plug
    "28:6c:07:87:54:b0": 9,  # mitu story teller
    "00:62:6e:51:27:2e": 10,  # Insteon Camera
    "a4:50:46:06:80:43": 10,  # xiaomi mobile
    "20:a6:0c:5a:42:10": 10,  # xiaomi tablet
    "28:3f:69:05:2d:b0": 10,  # sony mobile
    "44:80:eb:21:cb:95": 10,  # motorola mobile
}
Device2Stream = dict()#label:packet_embedding的列表
EmbeddingSize=11
TimeWindowSize=100#论文中的时间窗口，取100个数据包，相当于seq_len's=100

'''
packet_embedding[0]=dport
packet_embedding[1]=if ip
[2]=if tcp
[3]=if udp
[4]=if tls
[5]=if http
[6]=if dns
[7]=if other protocols
[8]=dir
[9]=packet size
[10]=time interval
'''
def my_cmp(x, y):
    if x[10] > y[10]:
        return 1
    if x[10] < y[10]:
        return 1
    return 0

def load_dataset():
    extensions=["eth.src", "eth.dst","frame.len","frame.time_epoch","ip","tcp","udp","tls","http","dns",
                "arp","ppp","icmp","ipv6","igmp","ah","esp","dccp","sctp","rtp","rtcp","smtp","ancp","dhcp","ftp","imap","nntp","ntp","pop","rsip","ssh","snmp","telnet","tftp","ssdp","mdns","quic","ndmp","its"]#目的端口（value.dport）在flowcontainer中默认提取了
    file_dir = r"../../../../../../DataSet/DataSet/IoT identification/Aisia CCS2020-Your smart home can't keep a " \
               r"secret Towards automated fingerprinting of iot traffic/iot-traffic-dataset"
    for file in getFiles(file_dir, '.pcapng'):
        packetlist = extract(infile=file, filter="", extension=extensions)
        for p in packetlist:
            packet_embedding=[0.0 for x in range(0,EmbeddingSize)]
            extension_list = p[10]  # flowcontainer/reader.py:10) extension(s)
            device_mac=""
            if extension_list[0] in MACAddress2Label.keys():#数据包p的源mac是设备之一，说明是出站包
                packet_embedding[8]=1.0#feature:dir
                device_mac=extension_list[0]
            else:#目的mac是设备，说明是入站++包
                device_mac = extension_list[1]
            packet_embedding[0]=float(p[8])#feature:dport
            packet_embedding[9]=float(extension_list[2])#feature:frame len
            packet_embedding[10]=float(extension_list[3])#feature:epoch time
            if extension_list[4]!='':
                packet_embedding[1]=1.0
            if extension_list[5]!='':
                packet_embedding[2]=1.0
            if extension_list[6]!='':
                packet_embedding[3]=1.0
            if extension_list[7]!='':
                packet_embedding[4]=1.0
            if extension_list[8]!='':
                packet_embedding[5]=1.0
            if extension_list[9]!='':
                packet_embedding[6]=1.0
            for i in range(10,len(extension_list)):#有一个满足就使packet_embedding[7]=1
                if extension_list[i]!='':
                    packet_embedding[7] = 1
            if device_mac not in Device2Stream:
                Device2Stream[device_mac] = []
            Device2Stream[device_mac].append(packet_embedding)#当前设备的数据包
    data=[]
    label=[]
    for device, pack_list in Device2Stream.items():
        pack_list.sort(key=functools.cmp_to_key(my_cmp))#按照epoch time 从小到大进行排序
        #compute time interval from the epoch time between two adjacent packets in one time window
        for num_window in range(0,min(5000,len(pack_list)//100)):#每个设备最多5000个窗口
            seq=[]
            for i in range(num_window*TimeWindowSize,num_window*TimeWindowSize+TimeWindowSize):
                if i==num_window*TimeWindowSize:
                    pack_list[i][10]=0.0
                else:
                    pack_list[i][10]-=pack_list[i-1][10]#计算和上一个包的间隔时间
                seq.append(pack_list[i])
            data.append(seq)
            label.append(MACAddress2Label[device])

    data = np.array(data)
    label = np.array(label)
    data = data.reshape(-1, EmbeddingSize)
    label = label.reshape(-1, 1)
    print("AsiaCCS2020 data's shape=", data.shape)
    print("AsiaCCS2020 label's shape=", label.shape)
    np.savetxt(r'../../data/AsiaCCS2020_data.csv', data, delimiter=',')
    np.savetxt(r'../../data/AsiaCCS2020_label.csv', label, delimiter=',')


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
