import unittest

import torch

import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torchvision as tv
import torch.utils.data
import matplotlib.pyplot as plt

from Learn import learn_sklearn, learn_pytorch


class TestSklearn(unittest.TestCase):
    def load_data(self,filename:str)->list:#从csv文件中读取数据，每行一个样本
        '''读取数据以后，统一的矩阵的每一行是一个样本，最后一列是标签(如有)'''
        dataset = []
        with open("../data"+filename, "r") as file:
            for i in file.readlines():
                a = i.strip().split(',')
                dataset.append([float(j) for j in a])
        return dataset
    def test_learn_preprocess(self):
        data = np.array(range(1, 10), dtype=np.float64).reshape(-1, 1)  # 生成9*1的矩阵，元素分别为[[1],[2],...[9]]
        learn_sklearn.learn_preprocess1(data)
        learn_sklearn.learn_preprocess2(data)
        learn_sklearn.learn_normalize(data)
        iris = datasets.load_iris()
        iris_data, iris_target = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size=0.3, random_state=42)
        print(X_train.size())

    def test_learn_linear_model(self):
        # 模拟数据
        x = np.linspace(0, 10, 50)
        noise = np.random.uniform(-2, 2, size=50)
        y = 5 * x + 6 + noise
        x_pred=[11]
        y_pred= learn_sklearn.learn_linear_model(np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1)), np.array(x_pred).reshape(-1, 1))
        print("线性回归预测",x_pred,"的预测结果是：",y_pred)

    def test_learn_LogisticRegression(self):
        data: list = self.load_data("test_LogisticRegression.txt")
        data=np.array(data)
        y=np.array(data[:,-1],dtype=int)#类别要求是int，强转
        y_pred= learn_sklearn.learn_LogisticRegression(data[:, :-1], y, np.array([[0.317029, 14.739025]]).reshape(1, -1))
        print("待分类数据属于：",y_pred)

    def test_learn_svc(self):
        iris=sklearn.datasets.load_iris()
        data=iris.data[:,2:4]
        label=(iris["target"] == 2).astype( np.float64 )
        learn_sklearn.learn_svc(data, label, np.array([[5.5, 1.7]]), np.array([[1]]))


class TestPytorch(unittest.TestCase):
    def test_learn_autograd(self):
        learn_pytorch.learn_autograd()
    def test_learn_partial_derivative(self):
        learn_pytorch.learn_partial_derivative()
    def test_learn_LinearRegression(self):
        learn_pytorch.learn_LinearRegression()
    def test_mynetmodel(self):
        torch.device('cuda')  # 使用cuda设备
        learn_pytorch.train_mynet1(torch.tensor([[8], [10]], dtype=torch.float32))#如果原始列表元素不写成.0的形式，则必须指定dtype=torch.float32
    def test_mynetmodel2(self):
        x_data = torch.tensor([#最外层的[]表示这是一个张量
            [1.0,1.0],   #这是一个样本
            [2.0,2.0],
            [3.0,3.0],
            [4.0,4.0],
            [5.0,5.0],
            [6.0,6.0],
            [7.0,7.0],
            [8.0,8.0],
            [9.0,9.0],
            [10.0,10.0]
                         ])*0.1
        '''
        这里，如果数据集是原来的[1,10],神经网络预测结果是Nan,而数据范围在[0,1]之间，能比较好地预测出来，
        可见数据归一化的重要性
        '''
        x_data1=x_data*x_data
        W=torch.tensor([[1.0],
                        [2.0]])#两个系数
        y_data=torch.matmul(x_data1,W)#torch.matmul做的是矩阵乘法，要求前一个矩阵的列数等于后一个矩阵的行数，拟合方程是x1^2+2*x2^2
        learn_pytorch.train_MyNetModel2(x_data, y_data, torch.tensor([[1.0, 1.0]]))#真实值是3，预测值2.8243


    def test_MyBPNet(self):
        torch.device('cuda')  # 使用cuda设备
        data=datasets.load_iris()
        iris_data = data['data']
        iris_type = data['target']
        iris_data= learn_sklearn.learn_preprocess2(iris_data)#先把每一个特征都归一化到[0,1]
        X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_type, test_size=0.3, random_state=52)#7:3的比例随机切分
        X_train=torch.FloatTensor(X_train)#训练数据是float
        y_train=torch.LongTensor(y_train)#标签
        X_test=torch.FloatTensor(X_test)
        y_test=torch.LongTensor(y_test)
        pre_test= learn_pytorch.train_MyBPNet(X_train, y_train, X_test)
        res=0.0
        for i in range(y_test.size()[0]):#y_test.size()[0]就是测试集样本个数
            if pre_test[i]==y_test[i]:
                res+=1
        print("分类准确率：",res/y_test.size()[0])

    def test_MyLetNet5(self):#获取MNIST的训练集和测试集
        #MNIST数据集，每个样本是3通道28*28的图片
        kwargs = {'num_workers': 2, 'pin_memory': True}
        train_dataset=tv.datasets.MNIST(root="../data",
                                        train=True,#是train还是test
                                        download=True,#是否下载
                                        transform=tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize((0.1307,), (0.3081,))])#MNIST的均值和标准差
                                        )#定义训练数据集并预处理
        #使用dataloader迭代器来加载数据集
        my_batch_size = 256  # 一次训练的样本个数
        train_loader=torch.utils.data.DataLoader(train_dataset,#数据集
                                                 batch_size=my_batch_size,#批训练大小
                                                 shuffle=True,#打乱数据
                                                 **kwargs
                                                 )
        test_loader=torch.utils.data.DataLoader(tv.datasets.MNIST(root="../data",
                                                                  train=False,
                                                                  download=True,
                                                                  transform=tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize((0.1307,), (0.3081,))])
                                                                  ),
                                                batch_size=my_batch_size,
                                                shuffle=True,
                                                **kwargs
                                                )
        #打印训练集的第一个图片
        plt.figure()
        img=tv.utils.make_grid(train_dataset.data[0]).numpy()#取训练集的data中的第一张图片
        plt.imshow(np.transpose(img,(1,2,0)))#原始维度是 (1, 28, 28) 的，需要转换维度为(28,28,1)
        plt.show()

        print("train_loader's shape",train_loader)
        learn_pytorch.train_mylenet5(train_loader, test_loader)

    def test_AlexNet(self):
        pass

    def test_mylstm(self):
        BATCH_SIZE = 3#每次训练是3个样本
        SEQ_LEN = 5#每个样本就相当于一句话，它里面有5个单词
        N_features = 10#每个单词用10个值元素的向量表示
        N_TrainData=BATCH_SIZE*10#训练集总数目

        X = torch.randn(N_TrainData, SEQ_LEN, N_features)#生成训练集，[样本数，序列数，embedding的维度]
        Y=torch.tensor(np.random.randint(0,1,size=[N_TrainData]))#二分类问题，标签就两个值
        '''
        一共30个样本，每个样本包含5个embedding，每个embedding是个10维向量，这30个样本分成10个batch，每个batch的size是3
        '''
        print("train_loader's shape", X.size(),"------- label's shape",Y.size())

        learn_pytorch.train_mylstm(X, Y, BATCH_SIZE)


'''
torchvision.transforms.Compose([操作1,...]):
图像预处理函数
Compose里面的参数实际上就是个列表，而这个列表里面的元素就是你想要执行的transform操作,操作按照列表元素的顺序被依次执行
常用的操作：
transforms.ToTensor()转换一个PIL库的图片或者numpy的数组为tensor张量类型，转换从[0,255]->[0,1]，其中[0,255]是灰度图像像素取值范围

transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))，通过平均值和标准差来标准化一个tensor图像，由于ToTensor()已经将图像变为[0,1]，我们使其变为[-1,1]，服从正态分布
第一个(0.5,0.5,0.5) 即三个通道的平均值
第二个(0.5,0.5,0.5) 即三个通道的标准差值
灰度图像的话是一个通道
'''

'''
DataLoader返回的是所有的数据，只是分成了每批次为batch_size大小的数据
DataLoader的shuffle参数，True 决定了是否能多次取出batch_size，False，则表明只能取出数据集大小的数据。
'''