#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as f
# torch.nn.functional必须在这里import，因为是个文件名而torch.nn.Module是个类
import numpy as np
import torch.utils.data


def learn_autograd():
    '''
    利用pytorch的自动求导机制实现的单变量函数的梯度下降法
    理解pytorc的自动求导机制，理解计算图模型，tensor的相关属性
    :return:
    '''
    torch.device('cuda')  # 使用cuda设备
    x = torch.tensor([0.0], requires_grad=True)
    print(x.shape)
    '''
    张量tensor里的值必须是float,可从python的list或者np.ndarray转换而来
    requires_grad=True才会自动计算该节点的导数
    x.data才能操作到张量x的数值
    '''
    y = pow(x, 2) + 2 * x + 1
    '''y关于x的导函数是y=2*x+2,可知当x=-1时y取最小值，现在要用梯度下降法迭代出最小值
    对于一元函数单变量只能在x轴方向上变化，所以导数为正，y递增，所以要减去
    '''
    '''反向求导函数保存y的梯度在叶子Tensor的grad属性中，只有叶子节点才会保存梯度值
    backward()调用一次，计算图模型就自动被释放，如果要针对y做多次求导，加参数retain_graph=True
    叶子节点的导数值每次计算完一次就要手动清零叶子节点名.grad.zero_()，否则再次求计算图的导数时会叠加求得的导数值
    '''
    alpha = 0.1  # 步长
    epochs = 50
    print("y about x's grad(epochs=:", epochs, "):", x.grad)
    for e in range(epochs):
        y.backward(retain_graph=True)  # 对当前x求导，因为要求多次导数所以保留当前计算图
        print("y about x's grad(epochs=:", e, "):", x.grad)
        x.data -= alpha * x.grad  # 梯度下降
        x.grad.zero_()  # 手动清零节点保存的梯度值，避免梯度值累加
    print("最终计算的最小值的解x=", x.data)


def learn_partial_derivative():
    '''
    求多元函数的偏导数，深入理解梯度的意义和计算图模型
    :return:
    '''
    W = torch.tensor([[1.0, 1.0], [2.0, 2.0]], requires_grad=False)  # 系数矩阵W
    x1 = torch.tensor([1.0], requires_grad=True)
    x2 = torch.tensor([5.0], requires_grad=True)
    print(W.shape, " ", x1.shape)
    u = W * x1
    y = u + x2  # 广播原则：程序会自动把 x2 扩展成 [[5.0, 5.0], [5.0, 5.0]]，和 u 的形状一样之后，再进行加法计算，
    loss = y.mean()  # 对矩阵求矩阵
    '''最终损失函数写成关于x1,x2的函数即：
    loss=1/4(1*x1+x2+1*x1+x2+2*x1+x2+2*x1+x2)
    '''
    print("loss=", loss.data)
    loss.backward()
    print(x1.data, x1.grad)  # dloss/dx1=1.5
    print(x2.data, x2.grad)  # dloss/dx2=1


def learn_LinearRegression():
    '''
    pytorch实现的y=w*x线性回归,每个数据全是标量
    :return:
    '''
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])  # 一行是一个样本,变量x有5个不同的取值，相当于5个样本
    noise = torch.randn(x.size())
    y = 5.0 * x + noise  # y的shape与x相同，每行是该样本的标签
    print(x.size(), " ", y.size())
    epochs = 50  # 迭代次数
    lr = 0.01  # 步长
    w = torch.tensor([0.0], requires_grad=True)  # 均方误差损失函数对参数w求导
    for i in range(epochs):
        y0 = x * w  # y的估计值
        loss = torch.mean((y0 - y) ** 2)  # 均方误差损失函数
        loss.backward()  # 计算该损失函数值关于参数w的导数
        # 梯度下降法更新参数
        w.data -= lr * w.grad
        w.grad.zero_()  # 手动清零节点保存的梯度值，避免梯度值累加
    print("线性回归估计的参数是：", w)


class MyNetModel1(torch.nn.Module):  # 自定义一个神经网络类模型，继承自torch.nn.Module
    def __init__(self):
        super(MyNetModel1, self).__init__()  # 调用父类torch.nn.Module的构造函数，此时应把网络组件传进去初始化
        self.predict = torch.nn.Linear(in_features=1, out_features=1, bias=True)  # 建立一个linear类,一个全连接层，它的特征维度是1，输出也是1

    def forward(self, X):  # 复写父类的forward方法,定义前向传播的过程,其中的数据X直接通过对象名(X)传进来就能调用了
        y_pred = self.predict(X)  # 单隐含层
        return y_pred


def train_mynet1(x_pred):
    mynetmodel = MyNetModel1()
    loss_mse = torch.nn.MSELoss(reduction="mean")  # 获得一个均方误差损失函数对象
    '''定义损失函数对象'''
    optimizer = torch.optim.SGD(mynetmodel.parameters(), lr=0.01)  # 设置利用SGD算法优化我的神经网络模型的参数
    '''构造一个optimizer对象。这个对象能保存当前的参数状态并且基于backward()计算出的梯度自动更新参数'''
    torch.manual_seed(2)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。如果设置初始化，则每次初始化都是固定的。
    x_data = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)  # 一行是一个样本
    y_data = torch.tensor([[3], [5], [7], [9], [11]], dtype=torch.float32)  # 每行对应它的对应标签
    # y=2*x+1
    epochs = 1000
    for e in range(epochs):
        y_pre = mynetmodel(x_data)  # 直接模型对象名(输入数据),它会自动调用forward函数得到预测数据
        loss = loss_mse(y_pre, y_data)  # 计算均方误差
        optimizer.zero_grad()  # 先将梯度清0再求，避免梯度累加到同一块内存
        loss.backward()  # 计算损失函数关于NN的参数的梯度
        optimizer.step()  # SGD更新参数
    print("神经网络参数集：", list(mynetmodel.parameters()))
    output = mynetmodel(x_pred)  # 用训练好的模型预测
    print("神经网络预测值：", output)


class MyNetModel2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        '''
        :param n_feature: 输入样本的特征数，注意不是样本数目，而是每个样本的特征数目
        :param n_hidden1: 第一层隐藏层的神经元个数
        :param n_hidden2:第二层隐藏层神经元的个数
        :param n_output: 输出层神经元的个数，例如二分类用sigmoid激活函数的话就是一个神经元
        '''
        super(MyNetModel2, self).__init__()
        self.hidden1 = torch.nn.Linear(in_features=n_feature, out_features=n_hidden1)  # 输入层-隐含层1
        self.hidden2 = torch.nn.Linear(in_features=n_hidden1, out_features=n_hidden2)  # 隐藏层1-隐藏层2
        self.predict = torch.nn.Linear(in_features=n_hidden2, out_features=n_output)  # 隐藏层2-输出层

    def forward(self, input_data):
        '''
        数据out一层一层地传递
        :param input_data: 在训练的时候MyNetModel2(X)即自动调用forward算法
        :return:
        '''
        out = self.hidden1(input_data)  # out:隐藏层1的输出
        out = f.relu(out)  # 将第一层隐含层的线性结果求relu激活函数
        out = self.hidden2(out)  # out:隐藏层2的输出
        out = f.relu(out)  # 将第二层隐含层的线性结果求relu激活函数
        out = self.predict(out)  # 输出层
        return out


def train_MyNetModel2(x_data, y_data, x_pre):
    '''
    :param x_data: 样本数目*特征数目大小的矩阵
    :param y_data: 标签矩阵
    :param x_pre: 待预测样本
    :return:
    '''
    netmodel2 = MyNetModel2(n_feature=2, n_hidden1=200, n_hidden2=200, n_output=1)  # 二元函数
    loss_mse = torch.nn.MSELoss(reduction="mean")  # 回归预测常使用均方误差损失函数
    optimizer = torch.optim.SGD(netmodel2.parameters(), lr=0.01)  # 设置利用随机梯度下降算法优化我的神经网络模型的参数
    torch.manual_seed(2)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。
    epochs = 10000
    # print(netmodel2)
    for e in range(epochs):
        y_pre = netmodel2(x_data)  # 自动调用forward函数
        loss = loss_mse(y_pre, y_data)  # 计算均方误差
        optimizer.zero_grad()  # 先将梯度清0再求，避免梯度累加到同一块内存
        loss.backward()  # 计算损失函数关于optimizer的梯度
        optimizer.step()  # SGD更新参数
    output = netmodel2(x_pre)  # 用训练好的模型预测
    print("神经网络预测值：", output)


class MyBPNet(torch.nn.Module):
    def __init__(self, n_fearures, n_hidden1, n_hidden2, n_hidden3, n_output):
        super(MyBPNet, self).__init__()
        self.hidden1 = torch.nn.Linear(in_features=n_fearures, out_features=n_hidden1)
        self.hidden2 = torch.nn.Linear(in_features=n_hidden1, out_features=n_hidden2)
        self.hidden3 = torch.nn.Linear(in_features=n_hidden2, out_features=n_hidden3)
        self.predict = torch.nn.Linear(in_features=n_hidden3, out_features=n_output)

    def forward(self, X):
        out = self.hidden1(X)
        out = f.relu(out)  # 将第一层的每个神经元的输出值用relu激活，维度不变
        out = self.hidden2(out)
        out = f.relu(out)
        out = self.hidden3(out)
        out = f.relu(out)
        out = self.predict(out)
        out = f.softmax(out)  # 三分类问题用softmax
        return out

def train_MyBPNet(x_data: torch.Tensor, y_data, x_pre):
    mybpnet = MyBPNet(n_fearures=x_data.size()[1], n_hidden1=1000, n_hidden2=1000, n_hidden3=1000, n_output=3)
    loss_cross = torch.nn.CrossEntropyLoss()  # 分类问题用交叉熵损失函数
    optimizer = torch.optim.SGD(mybpnet.parameters(), lr=0.01)  # 使用随机梯度下降法优化我的神经网络模型的参数
    torch.manual_seed(2)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。
    epochs = 10000
    for i in range(epochs):
        '''这里每一次训练都是一整个原始样本数据，没有划分batch_size'''
        y_pre = mybpnet(x_data)  # 自动调用forward函数
        loss = loss_cross(y_pre, y_data)  # 计算误差值
        optimizer.zero_grad()  # 先将梯度清0再求，避免梯度累加到同一块内存
        loss.backward()  # 计算损失函数关于optimizer的梯度
        optimizer.step()  # SGD更新参数
    output = mybpnet(x_pre)  # 用训练好的模型预测，输出维度是：待预测样本数*3
    prediction = torch.max(output, dim=1)[1]  # dim=1求output的每一行的最大值，[1]返回该最大值在out中的下标
    print("神经网络预测值：", prediction)
    return prediction


class MyLeNet5(torch.nn.Module):
    def __init__(self):
        super(MyLeNet5, self).__init__()  # 输入数据是单通道28*28
        '''两个卷积层'''
        self.conv1 = torch.nn.Conv2d(1, 6, (5, 5), padding=2,
                                     stride=1)  # 第一个卷积层,输入是灰度图片，因此单通道,第一个参数是1;输出是6个feature map，因为卷积核是5，padding=2，所以输出的每个特征图的大小=((28-5+2*2)/1)+1=28
        # 这之间的池化层在forward，经filter=(2,2),stride=2的池化，输出特征图的大小=((28-2+2*0)/2)+1=14
        self.conv2 = torch.nn.Conv2d(6, 16, (5, 5), stride=1)  # 第二个卷积层,输入6个feature map，输出16个通道，卷积核5*5，输出=(14-5)/1+1=10
        # 再经过一个(2,2)的池化，每个特征图=(10-2)/2+1=5
        '''定义最后三层全连接层'''
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # 第一层全连接层，输入是conv2经过池化以后得到的每个feature map是5*5的
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)  # 10个数字，输出是每个数字对应的概率

    def forward(self, x):  # 池化层在forward里
        print("origin data's shape",
              x.shape)  # --------------------------origin data's shape torch.Size([256, 1, 28, 28])，256是batch_size
        x = f.relu(self.conv1(x))
        print("conv1's output shape ",
              x.shape)  # --------------------------conv1's output shape  torch.Size([256, 6, 28, 28])
        x = f.max_pool2d(x, (2, 2), stride=2)  # 如果不显示指定stride的值，默认是filter的大小，注意这和Conv2d的不同，Conv2d的stride=的默认值是1
        print("maxpool1's output shape ", x.shape)
        # x先卷积，再relu激活，再池化，conv1输出的每个feature map的大小是28*28的，池化用和卷积一样的公式计算，所以池化后的map是14*14
        x = f.relu(self.conv2(x))
        print("conv2's output shape ", x.shape)
        x = f.max_pool2d(x, (2, 2), stride=2)  ##如果不显示指定stride的值，默认是filter的大小，注意这和Conv2d的不同，Conv2d的stride=的默认值是1
        print("maxpool2's output shape ", x.shape)
        '''
        x.size()返回值为(256, 16, 5, 5)，而池化后的大小是(16, 5, 5)，256是batch_size，一次训练样本的数目
        现在要把数据变成二维的，其中每一行是一个样本（即一张图片），只有二维张量才能送入全连接层
        '''
        x = x.view(x.size()[0], -1)  # 重新塑形,将多维数据重新塑造为二维数据，256*400，256是batch_size，400代表原来的一张图片的低维嵌入
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_mylenet5(TrainData: torch.utils.data.DataLoader, TestData: torch.utils.data.DataLoader):
    '''
    训练LeNet5的函数
    :param TrainData: 训练集数据
    :param TestData: 测试集数据
    :return:
    '''
    # 超参数
    use_gpu = torch.cuda.is_available()  # torch可以使用GPU时会返回true
    LR = 0.001  # 学习率
    epoch = 50
    model_lenet5 = MyLeNet5()  # 网络模型
    criterion_lenet5 = torch.nn.CrossEntropyLoss()  # 分类常用交叉熵损失
    optimizer_lenet5 = torch.optim.Adam(model_lenet5.parameters(), lr=LR, betas=(0.9, 0.999))  # 后两项时默认值
    if use_gpu:
        model_lenet5 = model_lenet5.cuda()  # 把网络模型放到gpu上
    else:
        pass  # 不显式指定的话默认使用cpu

    '''在使用pytorch构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是启用batch normalization和drop out'''
    model_lenet5.train()  # 训练CNN一定要调用.train()，测试的时候调用.eval()

    torch.manual_seed(2)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。
    for e in range(epoch):  # 每一次迭代都是针对全体数据做的,但是参数更新一batch一次
        sum_loss = 0.0
        for batch_index, (data, label) in enumerate(TrainData):  # 批数据的下标，该批数据的data及对应的label
            '''这里每次训练是一batch_size大小的样本数据集'''
            if use_gpu:
                data, label = data.cuda(), label.cuda()  # 把数据放到GPU上
            else:
                pass
            label_pre = model_lenet5(data)  # 送进去的训练数据的第一个维度是一batch_size大小,即每次训练的样本数目是一batch
            loss = criterion_lenet5(label_pre, label)  # 计算损失
            optimizer_lenet5.zero_grad()  # 梯度清零
            loss.backward()  # 计算梯度
            optimizer_lenet5.step()  # 更新网络参数
            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if batch_index % 100 == 99:
                print('[%d, %d] loss: %.03f'
                      % (e + 1, batch_index + 1, sum_loss / 100))  # 在第e次迭代第100个batch时的平均误差
                sum_loss = 0.0

    '''开始用TestData验证训练好的数据的准确率，测试的时候一定要model.eval()'''
    model_lenet5.eval()  # 测试的时候不要DropOut

    correct = 0
    total = 0
    for data in TestData:  # 每一批数据data
        images_test, labels_test = data
        if use_gpu:
            images_test, labels_test = images_test.cuda(), labels_test.cuda()  # 放到gpu上
        y_pre = model_lenet5(images_test)  # 预测，y_pre是batch_size*10的矩阵
        _, predicted = torch.max(y_pre, dim=1)  # 第一个值是具体的value（我们用下划线_表示），第二个值是value所在的index
        total += labels_test.size(0)  # 这一批的样本个数
        correct += (predicted == labels_test).sum().item()  # 这一批里预测和实际相同的样本个数
    print("识别准确率为:", (correct / total))


class MyLSTM(torch.nn.Module):
    '''
    参考资料：https://www.e-learn.cn/topic/3265111
    '''

    def __init__(self, n_features, n_hidden, n_layer, bid: bool, n_output,batch_size,seq_len):
        '''
        :param n_features: 特征维数，即一个word embedding的元素个数
        :param n_hidden:每个LSTM单元在某时刻t的输出的ht的维度
        :param n_layer:RNN层的个数：（在竖直方向堆叠的多个LSTM单元的层数）
        :param bid:是否是双向RNN，默认为false
        :param n_output:最后的全连接层的输出
        '''
        super(MyLSTM, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layer,bidirectional=bid,batch_first=True)
        # batch_first默认为False，但是通常应该置为True，这样数据维度的第0维就是表示样本，大小就是batch_size
        self.fc1 = torch.nn.Linear(in_features=n_hidden*seq_len, out_features=n_output)#线性层的输入是二维张量，其中第0维度一定是样本，第1维度才是每个样本的特征数

        self.batch_size=batch_size

    def forward(self, x):
        '''
        原始数据的形状=[batch_size,seq_len,n_features]
        输入数据的batch_size=3,seq_len=5,n_feature=10
        '''
        print("origin data's shape=",x.size())#--------------------[3,5,10]

        x, (hn, cn) = self.lstm1(x)
        '''
        lstm的输出的形状=[batch_size,seq_len,n_hidden]
        '''
        print("output size() of lstm1=",x.size())#---------------------[3,5,20]

        #print("before shape",x)
        x=x.reshape(self.batch_size,-1)#Linear的输入必须是二维张量,那就是一个样本，它的特征向量是：每个时间步的embedding的拼接
        #print(x)

        '''
        input shape of Linear=[batch_size,seq_len*n_hidden]
        '''
        print("input data's size() of Linear=", x.size())  # ---------------------[3,5*20]
        output=f.softmax(self.fc1(x),dim=1)

        return output

def train_mylstm(x_data: torch.Tensor, y_data: torch.Tensor,BATCH_SIZE:int):
    '''

    :param x_data: 全体训练数据，size=样本数*时刻数*每个时刻的特征数
    :param y_data: 训练数据对应的标签，size=样本数
    :param BATCH_SIZE:batch size，设置在DataLoader里，方便每次训练时取batch_size数目的样本
    :return:
    '''
    # 超参数
    use_gpu = torch.cuda.is_available()  # torch可以使用GPU时会返回true
    LR = 0.1  # 学习率
    epoch = 20
    SEQ_LEN=x_data.shape[1]#获取第1维度的数据，即每个样本的时刻数目，如果原来的数据集每个样本的时刻数目不一致，应该padding 0
    N_features=x_data.shape[2]#获取每个embedding的维度

    Trainloader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(x_data,y_data),  #！！！！！把训练集和其标签组合到一起！！！！！
        batch_size=BATCH_SIZE,  #batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
    )

    model_mylstm = MyLSTM(n_features=N_features,n_hidden=20,n_layer=2,bid=False,n_output=2,batch_size=BATCH_SIZE,seq_len=SEQ_LEN)  # 网络模型，n_layer层lstm+一层fc
    criterion = torch.nn.CrossEntropyLoss()  # 分类常用交叉熵损失
    optimizer= torch.optim.Adam(model_mylstm.parameters(), lr=LR, betas=(0.9, 0.999))  # 后两项时默认值
    if use_gpu:
        model_mylstm = model_mylstm.cuda()  # 把网络模型放到gpu上
    else:
        pass  # 不显式指定的话默认使用cpu

    '''在使用pytorch构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是启用batch normalization和drop out'''
    model_mylstm.train()  # 训练CNN一定要调用.train()，测试的时候调用.eval()

    torch.manual_seed(2)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。
    for e in range(epoch):  # 每一次迭代都是针对全体数据做的,但是参数更新一batch一次
        for batch_index, (data, label) in enumerate(Trainloader):  # 批数据的下标，该批数据的data及对应的label
            '''这里每次训练是一batch_size大小的样本数据集'''
            if use_gpu:
                data, label = data.cuda(), label.cuda()  # 把数据放到GPU上
            else:
                pass
            label_pre = model_mylstm(data)  # 送进去的训练数据的第一个维度是一batch_size大小,即每次训练的样本数目是一batch
            loss = criterion(label_pre, label)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新网络参数
            # 每次训练都打印该次batch的损失
            print('[%d, %d] loss: %.03f'
                  % (e + 1, batch_index, loss.item()))  # 在第e次迭代第100个batch时的平均误差


'''
卷积与池化输出维度计算公式：
    https://www.cnblogs.com/aaronhoo/p/12464941.html
'''

'''
torch.nn.Linear用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是 二维张量
Linear(输入的特征维度即输入矩阵的列数，
当前层输出矩阵的列数也即当前层神经元的个数，
bais:当前层有没有激活阈值b，默认True)：用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是二维张量

forward 是torch.nn.Module定义好的模板，表示前向传播，需要传给它样本数据,当执行model(x)的时候，底层自动调用forward方法计算结果
'''
'''
如何定义自己的网络：
需要继承nn.Module类，并实现forward方法。继承nn.Module类之后，在构造函数中要调用Module的构造函数, super(Linear, self).init()
一般把网络中具有可学习参数的层放在构造函数__init__()中。
不具有可学习参数的层（如激活层ReLU）可放在构造函数中，也可不放在构造函数中（此时在forward函数中使用nn.functional来代替）。
可学习参数放在构造函数中，并且通过nn.Parameter()使参数以parameters（一种tensor,默认是自动求导）的形式存在Module中，并且通过parameters()或者named_parameters()以迭代器的方式返回可学习参数。
只要在nn.Module中定义了forward函数，backward函数就会被自动实现（利用Autograd)。而且一般不是显式的调用forward(layer.forward), 而是layer(input), 会自执行forward()
'''

'''nn.Sequential : 把nn的层连接起来
我们发现每一层的输出作为下一层的输入，这种前馈nn可以不用每一层都重复的写forward()函数，
通过Sequential()和ModuleList()，可以自动实现forward。这两个函数都是特殊module, 包含子module。ModuleList可以当成list用，但是不能直接传入输入。
nn.Sequential()一个有顺序的容器，将特定神经网络模块按照在传入构造器的顺序依次被添加到计算图中执行。
'''
''' 
当卷积层步长为1时：原图片的大小-卷积核的大小+1+2*padding的值就是该卷积核对应的feature map的大小
例如：28-5+1+2*2=28,第一层卷积层的每个feature map的大小就是28*28
nn.Conv2d(in_channels=1, #输入通道，黑白图片就是1，RGB三原色就是3
                     out_channels=16, #输出的通道数目
                     kernel_size=(5,5), #卷积核的大小
                     stride=1, #filter step
                     padding=2 #设置在输入的所有边界增加值为 0 的边距的大小（也就是在feature map 外围增加几圈 0 ），
                     例如当 padding =1 的时候，如果原来大小为 3 × 3 ，那么之后的大小为 5 × 5 。即在外围加了一圈 0 
                    ), #output shape (16,28,28)
                    根据上面的公式可知，该Conv2d的输出特征图的大小=（28-5+2*2）/1+1=28
                    
nn.MaxPool2d(kernel_size=(2,2),#表示做最大池化的窗口大小，可以是单个值，也可以是tuple元组,
            stride=2, #确定这个窗口如何进行滑动。如果不指定这个参数，那么默认步长跟最大池化窗口大小一致。如果指定了参数，那么将按照我们指定的参数进行滑动。
                      #例如 stride=(2,3) ， 那么窗口将每次向右滑动三个元素位置，或者向下滑动两个元素位置。
)#该池化层输出特征图的大小=(28-2+0*2)/2+1=14
    注意：stride:如果不指定这个参数，那么默认步长跟最大池化窗口大小一致。如果指定了参数，那么将按照我们指定的参数进行滑动。
                    
                    
nn.ReLU(inplace=True), #inplace对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量

Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了。
但在测试及验证中：每个神经元都要参加运算，但其输出要乘以概率p
'''

'''
BATCH_SIZE的含义
BATCH_SIZE:即一次训练所抓取的数据样本数量;
若BATCH_SIZE=data.size[0](训练集样本数量);相当于直接抓取整个数据集，训练时间长，但梯度准确。
但不适用于大样本训练，比如IMAGENET。只适用于小样本训练，但小样本训练一般会导致过拟合现象，因此不建议如此设置。
'''
'''
Batch Normalization (BN) 就被添加在每一个全连接和激励函数之间
'''

'''RNN:
见图：https://img-blog.csdnimg.cn/20200113155845610.png
lstm输入是input, (h_0, c_0)
input shape:(seq_len, batch, input_size) 时间步数或序列长度，batch数，输入特征维度。如果设置了batch_first，则batch为第一维。
(h_0, c_0) 隐层状态
    h0 shape：(num_layers * num_directions, batch, hidden_size) 若是双向RNN则num_directions=2，否则num_directions=1
    c0 shape：(num_layers * num_directions, batch, hidden_size) 若是双向RNN则num_directions=2，否则num_directions=1

lstm输出是output, (h_n, c_n)
output shape:(seq_len, batch, hidden_size * num_directions) 包含每一个时刻的输出特征ht，如果设置了batch_first，则batch为第一维，
    output: 如果num_layer为3，则output只记录最后一层(即，第三层)的输出，shape为(time_step, batch, hidden_size * num_directions)， 包含每一个时刻的输出特征，与多少层无关。
    所以整体的LSTM的输出是在最后一个time_step时才能得到，才是最完整的最终结果，output最后一个时步的输出output[-1,:,:]
(h_n, c_n) 隐层状态 各个层的最后一个时步的隐含状态，所以与序列长度seq_len是没有关系的
    h_n shape: (num_layers * num_directions, batch, hidden_size)
    c_n shape: (num_layers * num_directions, batch, hidden_size)
'''
