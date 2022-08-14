#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def learn_preprocess1(data:np.ndarray)->np.ndarray:#data:样本数*特征数，即每一行是一个样本
    '''数据预处理: 基于mean和std的标准化
    标准化数据通过减去均值然后除以方差（或标准差），
    这种数据标准化方法经过处理后数据符合标准正态分布，即均值为0，标准差为1，
    '''
    ss=preprocessing.StandardScaler()
    scaler=ss.fit(data)#这一步可以得到scaler，scaler里面存的有计算出来的均值和方差
    print("每个特征的平均值：",ss.mean_)
    print("每个特征的方差：",ss.scale_)
    data=scaler.transform(data)  # 这一步再用scaler中的均值和方差来转换X，使X标准化
    print("标准化后的数据：",data)
    return data

def learn_preprocess2(data:np.ndarray)->np.ndarray:#data:样本数*特征数，即每一行是一个样本
    '''
    数据预处理：将每个特征值归一化到一个固定范围
    sklearn中的这个归一化是对列进行归一化，因此一列是一个特征
    '''
    scaler=preprocessing.MinMaxScaler(feature_range=(-1,1)).fit(data)
    data=scaler.transform(data)
    print("归一化到[-1,1]之后的数据：",data)
    return data

def learn_normalize(data:np.ndarray):#一行是一个样本
    '''首先求出样本的p-范数，然后该样本的所有元素都要除以该范数，
    这样最终使得每个样本的范数都为1
    '''
    data=preprocessing.normalize(data,norm='l2')#二范式，样本各个特征值除以各个特征值的平方之和
    print("正则化后的数据：",data)
def learn_onehot(data:np.ndarray):
    encoder=preprocessing.OneHotEncoder().fit(data)
    data=encoder.transform(data).toarray()#转换成二维数组
    return data
def learn_data_split(X:np.ndarray,y:np.ndarray,test_size:float,random_state:int):
    '''
    将数据集划分成训练集和测试集
    :param :X和y分别是所要划分的原始样本数据和对应的标签
    :return:X_train,X_test,y_train,y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=10)
    '''
    # train_data：所要划分的样本特征集
    # train_target：所要划分的样本结果
    # test_size：样本占比，如果是整数的话就是样本的数量
    # random_state：是随机数的种子,种子固定，使实验可复现
    '''
    return X_train,X_test,y_train,y_test

def learn_linear_model (X:np.ndarray,y:np.ndarray,X_pred:np.ndarray)->np.ndarray:
    '''线性回归
    一行是一个样本，数据是调用.reshape(-1, 1)传进来的
    '''
    from sklearn.linear_model import LinearRegression
    liner_model=LinearRegression(fit_intercept=True, normalize=False,
    copy_X=True, n_jobs=1)#创建模型
    liner_model.fit(X,y)#做拟合
    #liner_model.coef_是线性模型的系数，liner_model_intercept是截距
    print(liner_model)
    y_pred=liner_model.predict(X_pred)
    print("系数向量：",liner_model.coef_)
    print("截距：",liner_model.intercept_)
    '''作出数据散点图和拟合直线'''
    plt.figure(figsize=(5,5))
    plt.scatter(X,y)
    plt.plot(X,liner_model.predict(X),color="r")#作出拟合曲线
    plt.show()
    return y_pred

def learn_LogisticRegression(X:np.ndarray,y:np.ndarray,X_pred:np.ndarray):
    '''
    逻辑回归：实际上是分类器
    :param X: 一行是一个样本，
    :param y: 标签
    :return: 分类结果
    '''
    from sklearn.linear_model import LogisticRegression
    logistic_model=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
    fit_intercept=True, intercept_scaling=1, class_weight=None,
    random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
    verbose=0, warm_start=False, n_jobs=1)
    logistic_model.fit(X,y)
    print("分类器的系数：",logistic_model.coef_)
    print("分类器的截距：",logistic_model.intercept_)
    return logistic_model.predict(X_pred)

def learn_svc(X:np.ndarray, y:np.ndarray, x_test:np.ndarray,label_test:np.ndarray):
    '''
    二分类器
    :param X:
    :param y:
    :return:
    '''
    from sklearn.svm import SVC#svm用于分类
    SVC_model=SVC(C=1.0, kernel='rbf', gamma='auto')
    #C越大，越不能容忍错误分类，使用高斯核函数
    # gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，
    # gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。
    SVC_model.fit(X,y)#训练分类器
    print("支持向量机二分类结果是：", SVC_model.predict(x_test))
    print("支持向量机二分类的分数是：",SVC_model.score(x_test,label_test))




'''
list转换成ndarray,注意使用的是np.array(list)而不是np.ndarray()
'''
'''
fit函数仅返回相关信息，要转化data还得调用transform
'''
'''
sklearn为所有模型提供了非常相似的接口，这样使得我们可以更加快速的熟悉所有模型的用法。在这之前我们先来看看模型的常用属性和功能：

# 拟合模型
model.fit(X_train, y_train)
# 模型预测
model.predict(X_test)
# 获得这个模型的参数
model.get_params()
# 为模型进行打分
model.score(X_test, y_test)
'''