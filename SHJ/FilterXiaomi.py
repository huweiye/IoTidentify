#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json
import os
import re
import traceback

import requests
from flowcontainer.extractor import extract
from SHJ import utils

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
        print(res.text)
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
                for line in file_obj.readlines():
                    line=str(line)
                    if line.find(SRCMAC)!=-1:
                        srcMac= line[line.find(SRCMAC) + len(SRCMAC) + 3:line.find(SRCMAC) + len(SRCMAC) + 3 + 17]
                        if srcMac in visitedMac:
                            continue
                        if isXiaomi(macURL +srcMac):  # 源mac是小米设备
                            XiaomiMacSet.add(srcMac)
                        visitedMac.add(srcMac)
                    if line.find(DSTMAC)!=-1:
                        dstMac=(line[line.find(DSTMAC) + len(DSTMAC) + 3:line.find(DSTMAC) + len(DSTMAC) + 3 + 17])
                        if dstMac in visitedMac:
                            continue
                        if isXiaomi(macURL +dstMac):  # 源mac是小米设备
                            XiaomiMacSet.add(dstMac)
                        visitedMac.add(dstMac)
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
