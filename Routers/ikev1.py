#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from flowcontainer.extractor import extract
import json
import collections
import sys

def need_swap(index:int, payload_lengths:list)->bool: #查找该扩展里的第index个包
    for i,t in enumerate(payload_lengths):
        if i==index:#找到了我要找的那个数据包
            if t<0:
                return True
            else:
                return False

def get_pkt(index:int,one_extend:list)->tuple:#去流的扩展里找指定的数据包
    for i, t in enumerate(one_extend):
        if i==index:
            return t
#按包级别
def ikev1(result,extensiondict:list):
    for key in result: #遍历pcap文件里的每一条流
        value=result[key]
        pkts = [collections.OrderedDict()] * len(value)  # [包序号]:该包的json
        resvre_flag = False
        for i in range(len(pkts)):
            # 构造第i个包的json字符串
            pkts[i]['src'] = value.src
            pkts[i]['dst'] = value.dst
            pkts[i]['sport'] = value.sport
            pkts[i]['dport'] = value.sport
            pkts[i]['timestamps']=value.ip_timestamps[i]
            for key in extensiondict:  # 遍历每一个感兴趣键
                if key=="isakmp.exchangetype":
                    for key_tuple in value.extension[key]: #找当前流中包的序号为i的tuple
                        if key_tuple[1]==i: #找到了我要的那个包,取它前面的值
                            if int(key_tuple[0]) == 2:
                                pkts[i][key]="Main mode"
                            elif int(key_tuple[0]) == 32:
                                pkts[i][key] ="Quick mode"
                            elif int(key_tuple[0]) == 5:
                                pkts[i][key] ="Informational"
                            elif int(key_tuple[0]) == 4:
                                pkts[i][key] ="Aggressive"
                            else:
                                pkts[i][key] ="Unknown"
                            break #找到了我要找的包，退出循环找下一个关键字
                elif key=="isakmp.flags": #去找rspi
                    for _ ,key_tuple in enumerate(value.extension['isakmp.rspi']):
                        if key_tuple[1]==i: #我要找的包
                            if pkts[i]["isakmp.exchangetype"]=="Main mode":
                                if int(key_tuple[0], 16) == 0:
                                    pkts[i][key]="c2s_1"
                                    # 查看当前包的来源与去处
                                    if need_swap(_, value.payload_lengths):
                                        resvre_flag = True
                                else:  # 可能是main mode的第2，3，4，5，6个包
                                    # print(get_pkt(_,value.extension["isakmp.nextpayload"])[0])
                                    if '1' in get_pkt(_, value.extension["isakmp.nextpayload"])[0].split(','):
                                        pkts[i][key]="s2c_2"
                                    elif '4' in get_pkt(_, value.extension["isakmp.nextpayload"])[0].split(','):
                                        if need_swap(_, value.payload_lengths):  # S->C
                                            if resvre_flag:
                                                pkts[i][key]="c2s_3"  # C->S
                                            else:
                                                pkts[i][key]="s2c_4"  # S->C
                                        else:  # C->S
                                            if resvre_flag:
                                                pkts[i][key]="s2c_4"
                                            else:
                                                pkts[i][key] ="c2s_3"
                                    else:
                                        if need_swap(_, value.payload_lengths):  # S->C
                                            if resvre_flag:
                                                pkts[i][key]="c2s_5"  # C->S
                                            else:
                                                pkts[i][key]="s2c_6"  # S->C
                                        else:  # C->S
                                            if resvre_flag:
                                                pkts[i][key]="s2c_6"
                                            else:
                                                pkts[i][key]="c2s_5"
                            elif pkts[i]["isakmp.exchangetype"] == "Aggressive":
                                if int(key_tuple[0], 16) == 0:
                                    pkts[i][key]="c2s_1"
                                    # 查看当前包的来源与去处
                                    if need_swap(_, value.payload_lengths):
                                        resvre_flag = True
                                else:  # 可能是aggressive的第2,3个包
                                    if need_swap(_, value.payload_lengths):  # S->C
                                        if resvre_flag:
                                            pkts[i][key]="c2s_3"  # C->S
                                        else:
                                            pkts[i][key]="s2c_2"  # S->C
                                    else:  # C->S
                                        if resvre_flag:
                                            pkts[i][key]="s2c_2"
                                        else:
                                            pkts[i][key]="c2s_3"
                            else:  # 只能判断方向，不能判断是该序列的第几个包
                                if need_swap(_, value.payload_lengths):  # S->C
                                    if resvre_flag:
                                        pkts[i][key]="c2s_"  # C->S
                                    else:
                                        pkts[i][key]="s2c_"  # S->C
                                else:  # C->S
                                    if resvre_flag:
                                        pkts[i][key]="s2c_"
                                    else:
                                        pkts[i][key]="c2s_"
                            break
                else:#其他key
                    if key not in value.extension:
                        pkts[i][key]=None
                        continue
                    flag=False
                    for key_tuple in value.extension[key]: #找当前流中包的序号为i的tuple
                        if key_tuple[1]==i:
                            flag=True
                            pkts[i][key]=key_tuple[0]
                            break
                    if flag==False:
                        pkts[i][key]=None

            print_json(pkts[i])

#按流级别
'''
def ikev1(result,extensiondict:list):
    for key in result:  # 遍历每一条流
        flow = collections.OrderedDict()
        value = result[key]
        flow['src'] = value.src
        flow['dst'] = value.dst
        flow['sport'] = value.sport
        flow['dport'] = value.dport
        flow['timestamps'] = value.ip_timestamps
        resvre_flag=False

        for key in extensiondict:  # 遍历每一个感兴趣的键值对
            flow[key]=[]
            if key=="isakmp.exchangetype":
                for t in value.extension[key]:
                    if int(t[0]) ==2:
                        flow[key].append("Main mode")
                    elif int(t[0])==32:
                        flow[key].append("Quick mode")
                    elif int(t[0])==5:
                        flow[key].append("Informational")
                    elif int(t[0])==4:
                        flow[key].append("Aggressive")
                    else:
                        flow[key].append("Unknown")
            elif key=="isakmp.flags":
                for _,t in enumerate(value.extension['isakmp.rspi']):
                    if flow["isakmp.exchangetype"][_] =="Main mode":
                        if int(t[0],16)==0:
                            flow[key].append("c2s_1")
                            #查看当前包的来源与去处
                            if need_swap(_,value.payload_lengths):
                                resvre_flag=True
                        else: #可能是main mode的第2，3，4，5，6个包
                            #print(get_pkt(_,value.extension["isakmp.nextpayload"])[0])
                            if '1' in get_pkt(_,value.extension["isakmp.nextpayload"])[0].split(','):
                                flow[key].append("s2c_2")
                            elif '4' in get_pkt(_,value.extension["isakmp.nextpayload"])[0].split(','):
                                if need_swap(_,value.payload_lengths):#S->C
                                    if resvre_flag:
                                        flow[key].append("c2s_3")#C->S
                                    else:
                                        flow[key].append("s2c_4")#S->C
                                else:#C->S
                                    if resvre_flag:
                                        flow[key].append("s2c_4")
                                    else:
                                        flow[key].append("c2s_3")
                            else:
                                if need_swap(_, value.payload_lengths):  # S->C
                                    if resvre_flag:
                                        flow[key].append("c2s_5")  # C->S
                                    else:
                                        flow[key].append("s2c_6")  # S->C
                                else:  # C->S
                                    if resvre_flag:
                                        flow[key].append("s2c_6")
                                    else:
                                        flow[key].append("c2s_5")
                    elif flow["isakmp.exchangetype"][_] =="Aggressive":
                        if int(t[0],16)==0:
                            flow[key].append("c2s_1")
                            #查看当前包的来源与去处
                            if need_swap(_,value.payload_lengths):
                                resvre_flag=True
                        else:  # 可能是aggressive的第2,3个包
                            if need_swap(_, value.payload_lengths):  # S->C
                                if resvre_flag:
                                    flow[key].append("c2s_3")  # C->S
                                else:
                                    flow[key].append("s2c_2")  # S->C
                            else:  # C->S
                                if resvre_flag:
                                    flow[key].append("s2c_2")
                                else:
                                    flow[key].append("c2s_3")
                    else: #只能判断方向，不能判断是该序列的第几个包
                        if need_swap(_, value.payload_lengths):  # S->C
                            if resvre_flag:
                                flow[key].append("c2s_")  # C->S
                            else:
                                flow[key].append("s2c_")  # S->C
                        else:  # C->S
                            if resvre_flag:
                                flow[key].append("s2c_")
                            else:
                                flow[key].append("c2s_")
            else:
                if key not in value.extension:
                    flow[key]=None
                    continue
                for t in value.extension[key]:
                    flow[key].append(t[0])

        print_json(flow)
'''


def print_json(pkt):
    #把一个collections.OrderedDict()包转json
    json_fd = json.dumps(pkt, separators=(',', ': '))
    print(json_fd)

if __name__ == '__main__':
    extensiondict=["isakmp.exchangetype","isakmp.flags","isakmp.length","isakmp.payloadlength","isakmp.prop.transforms",
                                "isakmp.Routers.attr.encryption_algorithm","isakmp.Routers.attr.hash_algorithm","isakmp.Routers.attr.group_description","isakmp.Routers.attr.authentication_method",
                                "isakmp.Routers.attr.life_duration","isakmp.vid_string"]
    result = extract(r"C:\DataSet\IoT identification\UNSW-TMC2018-Classifying IoT Devices in Smart Environments Using Network Traffic Characteristics\16-09-25.tar\16-09-25\16-09-25.pcap",
                     filter='isakmp',
                     extension=["isakmp.exchangetype", "isakmp.length", "isakmp.payloadlength",
                                "isakmp.prop.transforms",
                                "isakmp.Routers.attr.encryption_algorithm", "isakmp.Routers.attr.hash_algorithm",
                                "isakmp.Routers.attr.group_description", "isakmp.Routers.attr.authentication_method",
                                "isakmp.Routers.attr.life_duration", "isakmp.vid_string",
                                "isakmp.rspi", 'isakmp.nextpayload']
                     )#isakmp.flags用于给当前数据包打握手标签

    print("num of flow:",len(result))
    savedStdout = sys.stdout  # 保存标准输出流
    with open(r'C:\DataSet\IoT identification\UNSW-TMC2018-Classifying IoT Devices in Smart Environments Using Network Traffic Characteristics\16-09-25.tar\16-09-25\16-09-25.txt', 'w+') as file:
        sys.stdout = file  # 标准输出重定向至文件
        ikev1(result,extensiondict)
    sys.stdout = savedStdout  # 恢复标准输出流