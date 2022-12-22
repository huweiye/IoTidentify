#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import requests
from flowcontainer.extractor import extract
from SHJ import utils

XiaomiMacSet=set()
visitedMac=set()
macURL="http://macvendors.co/api/"

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

def getXiaomiMac(file:str):
    '''
    读取pcap文件，输出是小米设备的mac地址
    :param file: pcap文件路径
    :return: 将小米设备的mac输出到文件
    '''
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

def printXiaomiMac(dir:str):
    '''
    :param dir: 包含pcap的文件夹路径
    :return:
    '''
    for file in utils.getFiles(dir, '.pcapng'):
        getXiaomiMac(file)
    with open('Xiaomimac.txt', 'a+') as f:
        for mac in XiaomiMacSet:
            f.write(mac+"\n")
