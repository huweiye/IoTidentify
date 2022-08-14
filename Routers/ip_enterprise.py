#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import sys
import os

import json
import re
import collections
import pandas as pd


noenter_dict = {'F11-NE40Ex16A-195': 'HuaWei', 'RouterOS': 'Mitrotik', 'idrac': 'Dell'}
enterprises = ['Cisco', 'Hangzhou H3C Technologies', 'HP', 'SANGFOR', 'Mitrotik', 'Clavister', 'HuaWei',
               'Digital China Networks', 'Brother', 'Juniper Networks', 'Ruijie Networks', 'ZTE',
               'Cnlink', 'TOSHIBA', 'Canon', 'SYNNEX', 'DbAppSecurity']

enterpeise_ips = dict()
ciscoversion_ip = dict()

def print_json(pkt):
    #把一个collections.OrderedDict()包转json
    json_fd = json.dumps(pkt, separators=(',', ': '))
    print(json_fd)

def line2ip(line: str):
    return line.split(' ')[0]


def get_ciscoVersion_ips(line: str):
    index1=-1
    index2=-1
    if re.search(r'Cisco IOS Software[\s,]',line)!=None:
        index1=re.search(r'Cisco IOS Software[\s,]',line).end()
    elif re.search(r'Cisco Internetwork Operating System Software \\r\\n',line)!=None:
        index1=re.search(r'Cisco Internetwork Operating System Software \\r\\',line).end()
    if index1 == -1:
        return
    if re.search(r'Version.*,', line)==None:
        return
    index2 = re.search(r'Version[^,]*', line).end()
    version = line[index1 + 1:index2]
    #print(version)
    if version not in ciscoversion_ip:
        ciscoversion_ip[version]=list()
    ciscoversion_ip[version].append(line2ip(line))


def get_enterprist_ips() :
    fp = open("D:\Documents\shj\iie\Routers\snmp\snmp_enterprise\snmp_by_serIP\snmp_by_serIP.txt", "r",
              encoding='UTF-8')
    while True:
        line = fp.readline()
        if not line:
            break
        flag = False
        for e in enterprises:
            is_this_enter = re.findall(e, line, flags=re.IGNORECASE)  # 为了忽略大小写比较
            if len(is_this_enter) == 0:
                continue
            if e not in enterpeise_ips:
                enterpeise_ips[e] = list()
            flag = True
            enterpeise_ips[e].append(line2ip(line))
            if e == 'Cisco':
                get_ciscoVersion_ips(line)
        # 对于还没找到厂商的行，再比较字典
        if flag == False:
            for key, value in noenter_dict.items():
                if line.find(key) != -1:
                    if value not in enterpeise_ips:
                        enterpeise_ips[value] = list()
                    flag = True
                    enterpeise_ips[value].append(line2ip(line))
    fp.close()


def get_version(myip:str)->str:
    for version,ips in ciscoversion_ip.items():
        for ip in ips:
            if ip==myip:
                return version


ip_message=dict()
def ip_enters():#反转enterpeise_ips和ciscoversion_ip
    for enterprise,ips in enterpeise_ips.items():
        for ip in ips:
            message=enterprise
            if enterprise=='Cisco':
                if get_version(ip)!=None:
                    message+=" "+get_version(ip)
            ip_message[ip]=message

def print_csv(d:dict):


    import csvhandler


    fileName = r"D:\Documents\shj\iie\Routers\snmp\snmp_enterprise\ip_message.csv"
    ##保存文件
    with open(fileName, "w+") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in d.items():
            writer.writerow([key, value])








if __name__ == "__main__":
    get_enterprist_ips()
    #print_json(enterpeise_ips)
    #print_json(ciscoversion_ip)
    ip_enters()
    print_csv(ip_message)

