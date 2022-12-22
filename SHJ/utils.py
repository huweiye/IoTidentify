#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
def getFiles(dir, suffix):
    '''
    :param dir: 目录
    :param suffix: 文件后缀
    :return: 返回指定目录下的指定文件后缀的文件路径列表
    '''
    res = []
    for root, _ , files in os.walk(dir):  # 当前根,根下目录,目录下的文件列表
        for filename in files:
            name, suf = os.path.splitext(filename)  # 文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename))  # 把一串字符串组合成文件的绝对路径
    return res