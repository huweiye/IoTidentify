#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import copy
import sys
import os
from string import digits
from typing import List

from flowcontainer.extractor import extract
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn import preprocessing
import json

ip_oid_dict = dict()
oid_set = set()
oid_sored_list = list()
index_ip = dict()  # 特征矩阵的行号到ip的映射
colors = ['c', 'r','b', 'g', 'm', 'y', 'k', 'w']
types=['o','s','p','*','+','-']


def is_enter(oids: str, oid_preix: str) -> bool:  # 给你一个包里出现的所有oid，判断这个包是不是指定想要的设备
    oid_array = oids.split(',')
    for oid in oid_array:
        if oid.startswith(oid_preix):
            return True
    return False


def get_enterprises_oid(oid_preix: str):  # 构造样本：ip-这个ip下所有oid值作为属性
    for key in result:  # 遍历每一条流
        value = result[key]
        distip = value.dst
        if 'snmp.name' not in value.extension:  # snmp.name这个字段值没有在这个流里的任何一个pkt里出现
            continue
        for t in value.extension['snmp.name']:
            oids = t[0]
            if is_enter(oids, oid_preix):  # 是我想要找的设备，把该目的ip出现的所有oid加进去
                if distip not in ip_oid_dict:
                    ip_oid_dict[distip] = list()
                for oid in oids.split(','):
                    oid_set.add(oid)
                    ip_oid_dict[distip].append(oid)


def get_features(oidlist: list) -> list:  # 为每一个样本(ip)构造它对应的oids特征，特征定义为:当前oid的出现次数
    f = []
    for oid in oid_sored_list:  # 统计当前oid在这个样本oidlist中的出现次数
        f.append(oidlist.count(oid))
    return f


def tsne_version(data, km, K: int):  # 对高维data进行可视化
    from sklearn.manifold import TSNE
    t_sne = TSNE(n_components=2, init='random', random_state=177).fit(data)
    df = pd.DataFrame(t_sne.embedding_)
    df['labels'] = km.labels_
    df_list = []
    for i in range(K):  # 聚了K类
        df_list.append(df[df['labels']==i])
    import matplotlib.pyplot as plt
    plt.figure()
    j=0
    for i in range(K):#用不同的颜色表示不同数据
        color = i % len(colors)
        if color==0:
            j+=1
        plt.plot(df_list[i][0], df_list[i][1], colors[color]+types[j])

    plt.savefig("picture/plot_kmeans_" + str(K) + ".png")


def print_res(flag: str, clu_res: list):  # 打印聚类结果，ip簇,输入clu_res是样本标签向量
    print(flag + "--------------------------------")
    res = dict()
    for _, label in enumerate(clu_res):
        if label not in res:
            res[label] = []
        res[label].append(index_ip[_])
    sorted(res.items(), key=lambda d: d[0])
    for key, value in res.items():
        print(key, ":", value)


def KMCluster(data: np.ndarray, num_clu: int):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(data)
    datascale = scaler.transform(data)
    km = KMeans(n_clusters=num_clu, max_iter=30)
    km.fit(datascale)
    from sklearn.metrics import silhouette_samples, silhouette_score
    sc = silhouette_score(datascale, km.labels_)
    tsne_version(data, km,num_clu)
    print_res("KMeans,when K=" + str(num_clu), km.labels_)
    print("silhoutte_score=", sc)  # 打印轮廓系数，轮廓系数s的取值再-1,1之间，s越靠近1越好，可以通过循环测试K值，取最优轮廓系数选定K


def LEVELClu(data: np.ndarray):  # 层次聚类
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,12))
    plt.xlabel('sample')
    plt.ylabel('distance')
    import scipy.cluster.hierarchy as sch
    Z = sch.linkage(data, method='average', metric='euclidean')  # 计算两个组合数据点中的每个数据点与其他所有数据点的距离。将所有距离的均值作为两个组合数据点间的距离。
    p = sch.dendrogram(Z)
    plt.savefig('picture/plot_dendrogram.png')
    my_t = 10  # 层次聚类的阈值
    cluster_res = sch.fcluster(Z, t=my_t, criterion='distance')
    print_res("cluster.hierarchy,distance scalar t=" + str(my_t), cluster_res.tolist())

def print_json(pkt):
    #把一个collections.OrderedDict()包转json
    json_fd = json.dumps(pkt, separators=(',', ': '))
    print(json_fd)

def jhs():
    oid_list_printjson = sorted(list(oid_set))
    ip_oidnum_printjson=dict()#ip-oid出现次数字典
    for ip, oidlist in ip_oid_dict.items():
        ip_oidnum_printjson[ip]=dict()
        for oid in oid_list_printjson:  # 统计当前oid在这个样本oidlist中的出现次数
            ip_oidnum_printjson[ip][oid]=(oidlist.count(oid),oidlist.count(oid)/len(oidlist))

    oid_num=dict()#统计所有的oid及其对应的出现次数
    oid_sum=0
    for oidlist in ip_oid_dict.values():
        for oid in oidlist:
            if oid not in oid_num:
                oid_num[oid]=0
            oid_num[oid]+=1
            oid_sum+=1
    oid_frequency_printjson=dict()#计算每一个oid的出现频率
    for oid,num in oid_num.items():
        oid_frequency_printjson[oid]=(num,num/oid_sum)


    print(oid_list_printjson)
    print_json(ip_oidnum_printjson)
    print_json(oid_frequency_printjson)

def b_get_oid(exten, index):
    for t in exten:
        if t[1]==index:
            return t[0]

def b_getciscostr(result):
    ip_oidvar_pj=dict()
    for key in result:  # 遍历每一条流
        value = result[key]
        src= value.src
        if 'snmp.var-bind_str' not in value.extension:  # 这个字段值没有在这个流里的任何一个pkt里出现
            continue
        for t in value.extension['snmp.var-bind_str']:
            var_bind_str=t[0]
            if var_bind_str.find('Cisco')!=-1:
                oid=b_get_oid(value.extension['snmp.name'], t[1])
                if src not in ip_oidvar_pj:
                    ip_oidvar_pj[src]=list()
                ip_oidvar_pj[src].append({oid:var_bind_str})
    #print(ip_oidvar_pj)
    savedStdout = sys.stdout  # 保存标准输出流
    with open(r'C:\Users\29205\Desktop\Log\oid_important.txt', 'w+') as file:
        sys.stdout = file  # 标准输出重定向至文件
        print_json(ip_oidvar_pj)
    sys.stdout = savedStdout  # 恢复标准输出流



if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')

    cisco_preix = "1.3.6.1.4.1.9"
    file_dir = r"D:\Documents\shj\iie\Routers\snmp\test_snmp"
    for file in os.listdir(file_dir):
        file_pcap_path = file_dir + '\\' + str(file)
        result = extract(file_pcap_path,
                         filter='snmp',
                         extension=["snmp.name","snmp.var-bind_str"])#
        # print("num of flow:", len(result))
        get_enterprises_oid(cisco_preix)
        jhs()
        b_getciscostr(result)






    savedStdout = sys.stdout  # 保存标准输出流

    #ip_oid_file = open(r'D:\Documents\shj\iie\Routers\snmp\snmp_pcaps\Log\ip_oid.txt', 'w+')
    #sys.stdout = ip_oid_file  # 标准输出重定向至文件
    # 打印存储着<ip,该ip下出现的oid列表>的字典
    print(json.dumps(ip_oid_dict, separators=(',', ': ')))
    sys.stdout = savedStdout  # 恢复标准输出流
    #ip_oid_file.close()

    oid_sored_list = sorted(list(oid_set))  # oid字典序列表
    data = []
    index = 0
    for ip, oidlist in ip_oid_dict.items():  # 生成特征矩阵，data[i][j]表示样本i中oid_sored_list[j]这个oid出现的次数
        index_ip[index] = ip
        data.append(get_features(oidlist))
        index += 1

    data_np = np.array(data)

    feature_file = open(r'D:\Documents\shj\iie\Routers\snmp\snmp_pcaps\Log\feature_matrix.txt', 'w+')
    sys.stdout = feature_file  # 标准输出重定向至文件
    # 打印原始特征向量
    print("sample's num=", len(data), "feature's num=", len(data[0]))
    print(data_np)
    sys.stdout = savedStdout  # 恢复标准输出流
    feature_file.close()

    result_file = open(r'D:\Documents\shj\iie\Routers\snmp\snmp_pcaps\Log\result.txt', 'w+')
    sys.stdout = result_file  # 标准输出重定向至文件

    LEVELClu(data_np)  # 层次聚类，得到：层次树、阈值为10下的

    for k in range(2, 3):
        KMCluster(data_np, k)

    sys.stdout = savedStdout  # 恢复标准输出流
    result_file.close()


