#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json
import os
import requests
from flowcontainer.extractor import extract

XiaomiMacSet=set()
visitedMac=set()
macURL="http://macvendors.co/api/"
ADDRSTR= "AddrStr"
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
        print(url)
        return True
    else:
        return False

def getXiaomiMac(file:str,suff:str):
    '''
    读取pcap文件或者日志文件，输出是小米设备的mac地址
    :param file: pcap文件路径
    :param suff:文件扩展名
    :return: 将小米设备的mac输出到文件
    '''
    if suff==".pcap":
        extensions = ["eth.src", "eth.dst"]
        flowDict = extract(infile=file, extension=extensions)
        for key in flowDict:
            flow = flowDict[key]
            srcMacList = flow.extension['eth.src']
            dstMacList = flow.extension['eth.dst']
            if srcMacList[0][0] in visitedMac and dstMacList[0][0] in visitedMac:#已经检查过的mac
                continue
            if isXiaomi(macURL+srcMacList[0][0]):  # 源mac是小米设备
                XiaomiMacSet.add(srcMacList[0][0])
            elif isXiaomi(macURL+dstMacList[0][0]):  # 目的mac是小米设备
                XiaomiMacSet.add(dstMacList[0][0])
            visitedMac.add(srcMacList[0][0])
            visitedMac.add(dstMacList[0][0])
    else:#处理的是log文件
        with open(file,"rb") as file_obj:
            try:
                for streamStr in file_obj.readlines():#对于组里的log文件，一行就是一个json
                    streamDict = json.loads(streamStr)
                    if streamDict[SRCMAC] not in visitedMac:
                        print(streamDict[SRCMAC])
                        if isXiaomi(macURL + streamDict[SRCMAC]):  # 源mac是小米设备
                            XiaomiMacSet.add(streamDict[SRCMAC])
                        visitedMac.add(streamDict[SRCMAC])
                    if streamDict[DSTMAC] not in visitedMac:
                        print(streamDict[DSTMAC])
                        if isXiaomi(macURL + streamDict[DSTMAC]):  # 目的mac是小米设备
                            XiaomiMacSet.add(streamDict[DSTMAC])
                        visitedMac.add(streamDict[DSTMAC])
            except Exception as e:
                print(e,file)

def printXiaomiMac(dir:str):
    '''
    :param dir: 包含pcap的文件夹路径
    :return:
    '''
    for root, _, files in os.walk(dir):
        for filename in files:
            name, suf = os.path.splitext(filename)  # 文件名,文件后缀
            if name=="act_log":#只检查日志文件
                getXiaomiMac(os.path.join(root, filename),suf)
    with open('Xiaomimac.txt', 'a+') as f:
        for mac in XiaomiMacSet:
            f.write(mac+"\n")

if __name__ == '__main__':
    dir = r"D:\29205workspace\Goolgle下载\数据(0)"
    dirlist = os.listdir(dir)
    for name in dirlist:
        if os.path.isdir(os.path.join(dir, name)):
            printXiaomiMac(os.path.join(dir, name))
