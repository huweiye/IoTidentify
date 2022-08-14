#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy
from scapy.all import *
import sys
from flowcontainer.extractor import extract


def test():
    pcap_test = rdpcap(r"D:\Documents\shj\iie\Routers\test-data\2901_IOS 15.4(3)M3-2911_IOS 15.4(3)M3_ipsec v2(2901_1.1--2911_7.8).pcap")
    for pkt in pcap_test:
        if "ISAKMP" in pkt and pkt["ISAKMP"].version == 0x20:
            pkt['ISAKMP'].display()


def test1():
    x=0xf
    for i in range(x):
        print(i)
def test2():
    for i in range(10):
        print(i,sep=",",end="")
    print("\n",end="")

def test3():
    result = extract(r"D:\Documents\重要文件\iie\路由识别\ike-ycx\ike.ycx.pcap", filter='isakmp',
                     extension=["isakmp.nextpayload","isakmp.rspi","isakmp.typepayload","isakmp.exchangetype","isakmp.length","isakmp.payloadlength","isakmp.criticalpayload","isakmp.spisize","isakmp.prop.transforms",
                                "isakmp.tf.id.encr","isakmp.tf.id.prf","isakmp.tf.id.integ","isakmp.tf.id.dh",
                                "isakmp.key_exchange.dh_group","isakmp.vid_string","isakmp.enc.data"])
    for key in result:
        value = result[key]
        print(value.src,value.dst)
        print(value.payload_lengths)
        print(value.extension['isakmp.rspi'])
        print(value.extension['isakmp.typepayload'][0][0].split(','))


def test4():
    result = extract(r"D:\Documents\shj\iie\Routers\pcap\test\merge_ike.pcap",
                     filter='isakmp',
                     extension=["isakmp.rspi"])

    print("num of flow:", len(result))
    for key in result:  # 遍历每一条流
        value=result[key]
        for _, key_tuple in enumerate(value.extension['isakmp.rspi']):
            print(_,key_tuple)

def test5():
    result = extract(r"D:\Documents\shj\iie\Routers\snmp\snmp (2).pcap",
                     filter='snmp',
                     extension=["snmp.name","snmp.var-bind_str"])

    print("num of flow:", len(result))
    for key in result:  # 遍历每一条流
        value = result[key]
        if 'snmp.name' not in value.extension:#snmp.name这个字段没有在这个流里的任何一个pkt里出现
            continue
        for t in value.extension['snmp.var-bind_str']:
            print(t)

def test6():
    import matplotlib.pyplot as plt
    import numpy as np
    x=np.array([[1,2,3],[2,4,6]])
    np.random.seed(1)
    colors = np.random.rand(10)
    for i in range(10):
        plt.figure()
        plt.scatter(x[:,0], x[:,1],c='r')

def test7():
    file_dir = r"D:\Documents\shj\iie\Routers\snmp\test_snmp"
    for file in os.listdir(file_dir):
        file_pcap_path = file_dir + '\\' + str(file)
        result = extract(file_pcap_path,
                         filter='snmp',
                         extension=["snmp.name","snmp.var-bind_str"])
        for key in result:  # 遍历每一条流
            value = result[key]
            print(value.src)
            print(value.dst)
            print(value.payload_lengths)
            if 'snmp.var-bind_str' not in value.extension:  # 想要的字段没有在这个流里的任何一个pkt里出现
                continue
            print(value.extension['snmp.var-bind_str'])


def test8():
    pcap_test = rdpcap(
        r"D:\Documents\shj\iie\Routers\snmp\test_snmp\1.pcap")
    for pkt in pcap_test:
        if 'SNMPvarbind' not in pkt:
            continue
        for i in pkt['SNMPvarbind']:
           i.show()





if __name__ == '__main__':
    test7()
