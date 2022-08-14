#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import copy
import sys
import os

from flowcontainer.extractor import extract

import numpy as np

import json
import re
import collections

enterprise = "Cisco"  # 目前只取思科的设备
valuetypes = ["snmp.var-bind_str", "snmp.value.oid", "snmp.value.timeticks", "snmp.value.int"]

version_ip = collections.OrderedDict()  # 型号-ip
ip_varlist = collections.OrderedDict()  # ip-oidkey和value
ip_keylist = collections.OrderedDict()  # ip-oidkey


def get_pkt_src(index: int, payloads: list, src, dst):
    if payloads[index] > 0:
        return src
    else:
        return dst


def get_valuelist(index, exten,str):
    if str not in exten:
        return []
    for t in exten[str]:
        if t[1] == index:
            if str!="snmp.var-bind_str":
                oid_array = t[0].split(',')
                return oid_array
            else:
                return [t[0]]



def get_ip_enter(result):
    for key in result:  # 遍历每一条流
        value = result[key]
        src = value.src
        dst = value.dst
        payloads = value.payload_lengths
        if 'snmp.var-bind_str' not in value.extension:  # 想要的字段没有在这个流里的任何一个pkt里出现
            continue
        for t in value.extension['snmp.var-bind_str']:  # 对每一个pkt
            index = t[1]
            varstr = t[0]
            if varstr.find(enterprise) != -1:  # 包含'Cisco'字符串,是Cisco的设备
                version = re.search(r"C\d+[A-Z]*", varstr)  # 正则表达式取CXXX
                if version:
                    pkt_src = get_pkt_src(index, payloads, src, dst)
                    version_ip[version.group()] = pkt_src  # 记录路由器型号-它的ip
                    if pkt_src not in ip_varlist:
                        ip_varlist[pkt_src] = list()
                    # 找这个index的包里面所有的oid-value值对
                    for oidvalue in valuetypes:
                        valuelist=get_valuelist(index, value.extension, oidvalue)
                        if valuelist==None:
                            continue
                        for v in valuelist:
                            ip_varlist[pkt_src].append(v)
                    if pkt_src not in ip_keylist:
                        ip_keylist[pkt_src] = list()
                    for k in get_valuelist(index, value.extension,'snmp.name'):
                        ip_keylist[pkt_src].append(k)


def counter(l: list):
    d = collections.OrderedDict()
    for e in l:
        if e not in d:
            d[e]=0
        d[e] += 1
    return d

def print_json(pkt):
    #把一个collections.OrderedDict()包转json
    json_fd = json.dumps(pkt, separators=(',', ': '))
    print(json_fd)

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')
    file_dir = r"D:\Documents\shj\iie\Routers\snmp\snmp_pcaps\snmp_pcaps"
    for file in os.listdir(file_dir):
        file_pcap_path = file_dir + '\\' + str(file)
        result = extract(file_pcap_path,
                         filter='snmp',
                         extension=["snmp.name", "snmp.var-bind_str", "snmp.value.oid", "snmp.value.timeticks",
                                    "snmp.value.int"])
        get_ip_enter(result)

    IPKEY = collections.OrderedDict()
    for k, v in ip_keylist.items():
        IPKEY[k] = counter(v)

    IPVALUE=collections.OrderedDict()
    for k, v in ip_varlist.items():
        IPVALUE[k] = counter(v)

    savedStdout = sys.stdout  # 保存标准输出流
    with open(r'D:\Documents\shj\iie\Routers\snmp\snmp_pcaps\Log\Cisco_OID\ip_keyoid.txt', 'w+') as file:
        sys.stdout = file  # 标准输出重定向至文件
        print_json(IPKEY)
    sys.stdout = savedStdout  # 恢复标准输出流

    with open(r'D:\Documents\shj\iie\Routers\snmp\snmp_pcaps\Log\Cisco_OID\ip_value.txt', 'w+') as file:
        sys.stdout = file  # 标准输出重定向至文件
        print_json(IPVALUE)
    sys.stdout = savedStdout  # 恢复标准输出流

    with open(r'D:\Documents\shj\iie\Routers\snmp\snmp_pcaps\Log\Cisco_OID\version_ip.txt', 'w+') as file:
        sys.stdout = file  # 标准输出重定向至文件
        print_json(version_ip)
    sys.stdout = savedStdout  # 恢复标准输出流

