#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
from flowcontainer.extractor import extract
import json
import collections
import sys

shebeilist=set()

def print_shebei():
    for s in shebeilist:
        print(s)

def print_json(pkt):
    #把一个collections.OrderedDict()包转json
    json_fd = json.dumps(pkt, separators=(',', ': '))
    print(json_fd)

def snmp(result,enter_num:dict()):
    for key in result:  # 遍历每一条流
        value = result[key]
        flow = collections.OrderedDict()
        flow['src'] = value.src
        flow['dst'] = value.dst
        flow['sport'] = value.sport
        flow['dport'] = value.dport
        flow['timestamps'] = value.ip_timestamps
        if 'snmp.name' not in value.extension:
            continue
        for t in value.extension['snmp.name']:
            oid=t[0]
            obj_str="1.3.6.1.4.1."
            index_str=oid.find(obj_str)#这里逻辑有问题，因为一个pkt里有多个oid，它们组成了一个字符串t[0],不应该直接匹配，而是根据.分成字符串数组，对每一个数组元素做前缀匹配
            if index_str==-1:
                continue
            temp=oid[index_str+len(obj_str):]
            index_str=temp.find('.')
            num=temp[:index_str]
            flow[oid]=enter_num[int(num)]
            shebeilist.add(enter_num[int(num)])
        if len(flow)>5:
            print_json(flow)

def enternum()->dict():
    oids=dict()
    fp = open("D:\Documents\shj\iie\Routers\snmp\e.txt", "r",encoding='UTF-8')
    while True:
        line = fp.readline()
        if not line:
            break
        line = line.replace("\n", "")
        if line.isdigit():
            id=int(line)
            Organization = fp.readline().strip()#厂商
            oids[id]=Organization
    return oids



if __name__ == '__main__':
    enter_num=enternum()
    '''
    result = extract(r"D:\Documents\重要文件\iie\路由识别\snmp\snmp (2).pcap",
                     filter='snmp',
                     extension=["snmp.name"])

    
    savedStdout = sys.stdout  # 保存标准输出流
    with open(r'D:\Documents\重要文件\iie\路由识别\snmp\snmp (2).txt', 'w+') as file:
        sys.stdout = file  # 标准输出重定向至文件
        snmp(result,enter_num)
    sys.stdout = savedStdout  # 恢复标准输出流
    '''

    savedStdout = sys.stdout  # 保存标准输出流
    file_dir=r"D:\Documents\重要文件\iie\路由识别\snmp\snmp_pcaps\snmp_pcaps"
    for file in os.listdir(file_dir):
        file_pcap_path= file_dir + '\\' + str(file)
        result = extract(file_pcap_path,
                         filter='snmp',
                         extension=["snmp.name"])

        file_txt_path= file_pcap_path[:-4] + 'txt'
        print("print to:", file_txt_path)
        with open(file_txt_path, 'w+') as f:
            sys.stdout = f  # 标准输出重定向至文件
            snmp(result, enter_num)
        sys.stdout = savedStdout  # 恢复标准输出流

    with open(r'D:\Documents\重要文件\iie\路由识别\snmp\snmp_pcaps\enterprises.txt', 'w+') as file:
        sys.stdout = file  # 标准输出重定向至文件
        print_shebei()
    sys.stdout = savedStdout  # 恢复标准输出流

