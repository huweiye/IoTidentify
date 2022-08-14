#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from flowcontainer.extractor import extract
import json
import collections
import sys

#按包级别
def ikev2(result,extensiondict:list):
    for key in result:#遍历每一条流
        value = result[key]
        pkts=[collections.OrderedDict()]*len(value)#包序号-该包的json
        for i in range(len(pkts)):
            #构造第i个包的json字符串
            pkts[i]['src']=value.src
            pkts[i]['dst']=value.dst
            pkts[i]['sport']=value.sport
            pkts[i]['dport']=value.sport
            pkts[i]['timestamps']=value.ip_timestamps[i]
            for key in extensiondict:#遍历每一个感兴趣键
                if key not in value.extension:#这个包所在的流里没有这个键
                    pkts[i][key]=None
                else:
                    flag=False
                    for key_tuple in value.extension[key]: #找当前流中包的序号为i的tuple
                        if key_tuple[1]==i: #找到了我要的那个包,取它前面的值
                            flag=True
                            if key=="isakmp.exchangetype":
                                if int(key_tuple[0])==37:
                                    pkts[i][key] = "INFORMATIONAL"
                                elif int(key_tuple[0])==34:
                                    pkts[i][key] =  "IKE_SA_INIT"
                                elif int(key_tuple[0])==35:
                                    pkts[i][key] =  "IKE_AUTH"
                                elif int(key_tuple[0])==36:
                                    pkts[i][key] =  "CREATE_CHILD_SA"
                                else:
                                    pkts[i][key] = "UNKNOWN"
                            elif key=="isakmp.flags":
                                if pkts[i]["isakmp.exchangetype"]=="INFORMATIONAL":
                                    if key_tuple[0]=="0x00" or key_tuple[0]=="0x20":
                                        pkts[i][key]="s2c_"
                                    elif key_tuple[0]=="0x28" or key_tuple[0]=="0x08":
                                        pkts[i][key]="c2s_"
                                    else:
                                        pkts[i][key] ="unknown"
                                elif pkts[i]["isakmp.exchangetype"]=="IKE_SA_INIT":
                                    if key_tuple[0]=="0x08":
                                        pkts[i][key]="c2s_1"
                                    elif key_tuple[0]=="0x20":
                                        pkts[i][key]="s2c_2"
                                    else:
                                        pkts[i][key]="unknown"
                                elif pkts[i]["isakmp.exchangetype"]=="IKE_AUTH":
                                    if key_tuple[0] == "0x08":
                                        pkts[i][key] = "c2s_1"
                                    elif key_tuple[0] == "0x20":
                                        pkts[i][key] = "s2c_2"
                                    else:
                                        pkts[i][key] = "unknown"
                                elif pkts[i]["isakmp.exchangetype"]=="CREATE_CHILD_SA":
                                    if key_tuple[0] == "0x08" or key_tuple[0]=="0x28":
                                        pkts[i][key] = "c2s_"
                                    elif key_tuple[0] == "0x00" or key_tuple[0] == "0x20":
                                        pkts[i][key] = "s2c_"
                                    else:
                                        pkts[i][key] = "unknown"
                                else:
                                    pkts[i][key] = "unknown"
                            else: #其他key取提取出来的原值
                                pkts[i][key]=key_tuple[0]
                    if flag==False:
                        pkts[i][key]=None

            print_json(pkts[i])

#按流级别
'''
def ikev2(result,extensiondict:list):
    for key in result:  # 遍历每一条流
        flow=collections.OrderedDict()
        value = result[key]
        flow['src']=value.src
        flow['dst']=value.dst
        flow['sport']=value.sport
        flow['dport']=value.dport
        flow['timestamps']=value.ip_timestamps
        for key in extensiondict:  # 遍历每一个感兴趣的键值对
            if key not in value.extension:
                flow[key] = None
                continue
            flow[key]=[]
            if key=="isakmp.exchangetype":
                for t in value.extension[key]:
                    if int(t[0]) == 34:
                        flow[key].append("IKE_SA_INIT")
                    elif int(t[0]) == 35:
                        flow[key].append("IKE_AUTH")
                    elif int(t[0]) == 36:
                        flow[key].append("CREATE_CHILD_SA")
                    elif int(t[0]) == 37:
                        flow[key].append("INFORMATIONAL")
                    else:
                        flow[key].append("UNKNOWN")
            elif key=="isakmp.flags":
                for _,t in enumerate(value.extension[key]):
                    if flow["isakmp.exchangetype"][_] == "INFORMATIONAL":
                        if t[0] == "0x00" or t[0] == "0x20":
                            flow[key].append("s2c_")
                        elif t[0] == "0x28" or t[0] == "0x08":
                            flow[key].append("c2s_")
                        else:
                            flow[key].append("unknown")
                    elif flow["isakmp.exchangetype"][_] == "IKE_SA_INIT":
                        if t[0] == "0x08":
                            flow[key].append("c2s_1")
                        elif t[0] == "0x20":
                            flow[key].append("s2c_2")
                        else:
                            flow[key].append("unknown")
                    elif flow["isakmp.exchangetype"][_] == "IKE_AUTH":
                        if t[0] == "0x08":
                            flow[key].append("c2s_1")
                        elif t[0] == "0x20":
                            flow[key].append("s2c_2")
                        else:
                            flow[key].append("unknown")
                    elif flow["isakmp.exchangetype"][_] == "CREATE_CHILD_SA":
                        if t[0] == "0x08" or t[0] == "0x28":
                            flow[key].append("c2s_")
                        elif t[0] == "0x00" or t[0] == "0x20":
                            flow[key].append("s2c_")
                        else:
                            flow[key].append("unknown")
                    else:
                        flow[key].append("unknown")
            else:
                for t in value.extension[key]:
                    flow[key].append(t[0])
        print_json(flow)
'''
def print_json(pkt):
    #把一个collections.OrderedDict()包转json
    json_fd = json.dumps(pkt, separators=(',', ': '))
    print(json_fd)

if __name__ == '__main__':
    extensions=["isakmp.exchangetype", "isakmp.flags", "isakmp.length", "isakmp.payloadlength", "isakmp.prop.transforms",
                                "isakmp.tf.id.encr","isakmp.tf.id.prf","isakmp.tf.id.integ","isakmp.tf.id.dh",
                                "isakmp.key_exchange.dh_group","isakmp.vid_string","isakmp.notify.msgtype"]
    result = extract(r"D:\Documents\shj\iie\Routers\pcap\test\merge_ike.pcap",
                     filter='isakmp.version==0x20',
                     extension=["isakmp.exchangetype","isakmp.flags","isakmp.length","isakmp.payloadlength","isakmp.prop.transforms",
                                "isakmp.tf.id.encr","isakmp.tf.id.prf","isakmp.tf.id.integ","isakmp.tf.id.dh",
                                "isakmp.key_exchange.dh_group","isakmp.vid_string","isakmp.notify.msgtype"])#isakmp.flags用于给当前数据包打握手标签

    print("num of flow:",len(result))
    savedStdout = sys.stdout  # 保存标准输出流
    with open(r'D:\Documents\shj\iie\Routers\pcap\test\merge_ikev2.txt', 'w+') as file:
        sys.stdout = file  # 标准输出重定向至文件
        ikev2(result, extensions)
    sys.stdout = savedStdout  # 恢复标准输出流








    '''keylist = ['sip', 'dip', 'sport', 'dport', 'ISAKMP_exchange_type','ISAKMP_total_length',
               'ISAKMP_SA_length','ISAKMP_SA_critical_payload',
               'ISAKMP_proposal_SPIsize','ISAKMP_proposal_transform_number',
               'ISAKMP_transform_encr', 'ISAKMP_transform_prf','ISAKMP_transform_integ','ISAKMP_transform_dh',
               "ISAKMP_KeyExchange_dh_group","ISAKMP_VendorID","ISAKMP_notify_msgtype"]
    '''
