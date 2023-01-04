#!/usr/bin/env python 
# -*- coding:utf-8 -*-
#过滤ycx的小米设备用
import json
import os
import requests
from flowcontainer.extractor import extract

XiaomiMacSet=set()
VisitedMac=set()
macURL="http://macvendors.co/api/"
SRCMAC= "SrcMac"
DSTMAC= "DstMac"

def isXiaomi(url:str)->bool:
    '''
    判断该mac是否是小米的设备
    :param url:
    :return:是否是小米设备
    '''
    # 导入requests包
    res = requests.get(url=url)
    import re
    p = re.compile('XIAOMI', re.IGNORECASE)
    if p.search(res.text):#mac地址查厂商是小米的设备
        return True
    else:
        return False

def handleMac(srcMac:str,dstMac:str):
    if srcMac not in VisitedMac:
        if isXiaomi(macURL + srcMac):
            XiaomiMacSet.add(srcMac)  # 源mac是小米设备
        VisitedMac.add(srcMac)
    if dstMac not in VisitedMac:
        if isXiaomi(macURL + dstMac):  # 目的mac是小米设备
            XiaomiMacSet.add(dstMac)
        VisitedMac.add(dstMac)

def getXiaomiMac(file:str,suff:str):
    '''
    读取pcap文件或者log文件，输出是小米设备的mac地址
    :param file: pcap文件路径
    :param suff:文件扩展名
    :return: 将小米设备的mac输出到文件
    '''
    try:
        if suff==".pcap":
            extensions = ["eth.src", "eth.dst"]
            flowDict = extract(infile=file, extension=extensions)
            for key in flowDict:
                flow = flowDict[key]
                srcMacList = flow.extension['eth.src']
                dstMacList = flow.extension['eth.dst']
                srcMac=srcMacList[0][0]
                dstMac=dstMacList[0][0]
                handleMac(srcMac,dstMac)
        else:#处理的是log文件
            with open(file,"rb") as file_obj:
                for streamStr in file_obj.readlines():#组里的log文件，一行就是一个json
                    streamDict = json.loads(streamStr)
                    srcMac=streamDict[SRCMAC]
                    dstMac=streamDict[DSTMAC]
                    handleMac(srcMac, dstMac)
    except Exception as e:
        print("文件%s出错%s",file,str(e))

def printXiaomiMac(dir:str):
    '''
    :param dir: 包含pcap的文件夹路径
    '''
    for root, _, files in os.walk(dir):
        for filename in files:
            name, suf = os.path.splitext(filename)  # 文件名,文件后缀
            getXiaomiMac(os.path.join(root, filename),suf)

if __name__ == '__main__':
    dir = r"/data2/devicelog/pcap_dpkt"
    for subdir1 in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, subdir1)):
            dir1=os.path.join(dir, subdir1)
            for subdir2 in os.listdir(dir1):
                if os.path.isdir(os.path.join(dir1, subdir2)):
                    dir2=os.path.join(dir1, subdir2)
                    printXiaomiMac(dir2)
    with open('Xiaomimac.txt', 'a+') as f:
        for mac in XiaomiMacSet:
            f.write(mac+"\n")
    with open("AllMac.txt",'a+') as f:
        for mac in VisitedMac:
            f.write(mac+"\n")