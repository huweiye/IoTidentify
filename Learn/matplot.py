import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay



def plt_loss_and_acc():
    losslist = [2.437830686569214, 1.361871361732483, 1.2354837656021118, 0.8214618563652039, 0.8825271725654602, 0.7406708002090454, 0.7199279069900513, 0.6554309725761414, 0.5868269801139832, 0.6098504066467285,
                0.595007598400116, 0.5464728474617004,
                0.5358387231826782, 0.5703977942466736, 0.8878699541091919, 0.6499577164649963, 0.5103843212127686, 0.46383488178253174, 0.49370959401130676, 0.517888069152832, 0.6212297081947327, 0.598397433757782, 0.39335307478904724, 0.3683716952800751,
                0.5514708757400513, 0.5193508863449097,
                0.4998524487018585, 0.5277264714241028, 0.5942178964614868, 0.5157484412193298, 0.463947057723999, 0.47764497995376587, 0.6540297269821167, 0.4553811252117157, 0.520585834980011, 0.3755347430706024, 0.42438462376594543, 0.6017739176750183, 0.47218531370162964, 0.42659619450569153,
                0.39164823293685913, 0.4638076424598694, 0.4766503870487213, 0.3423151671886444, 0.24562503397464752, 0.46304988861083984, 0.4819560647010803, 0.4990854263305664, 0.357366681098938, 0.37246522307395935, 0.522031307220459, 0.46962881088256836, 0.5370092391967773, 0.4038323760032654, 0.6843863725662231, 0.41642168164253235, 0.3887793719768524,
                0.4125250577926636, 0.3002241849899292, 0.31478041410446167, 0.3916361927986145, 0.52137690782547, 0.33885657787323, 0.3977912664413452, 0.1719680279493332, 0.36743345856666565, 0.4664308726787567, 0.3075028359889984, 0.3377518057823181, 0.3716583549976349, 0.4272085726261139, 0.32078438997268677, 0.5404221415519714, 0.3062308132648468, 0.46064168214797974,
                0.3059377372264862, 0.33580660820007324, 0.3732936978340149, 0.43431058526039124, 0.3255392611026764, 0.3880864679813385, 0.30456918478012085, 0.3731056749820709, 0.315848171710968, 0.568324089050293, 0.536825954914093, 0.31702160835266113, 0.2787453830242157, 0.4410364627838135,
                0.3791040778160095, 0.2831626832485199, 0.39948731660842896, 0.39722514152526855]
    acclist = [0.109375, 0.5546875, 0.5859375, 0.765625, 0.7265625, 0.765625, 0.7578125, 0.78125,
               0.8203125, 0.78125, 0.796875, 0.8125, 0.8203125, 0.8046875, 0.71875, 0.78125, 0.8515625, 0.859375, 0.84375, 0.8046875, 0.8359375, 0.8046875, 0.859375, 0.890625, 0.8046875, 0.8203125, 0.8046875, 0.8671875, 0.828125, 0.84375, 0.8203125, 0.84375, 0.8359375, 0.890625, 0.828125, 0.875, 0.8515625, 0.8203125, 0.84375, 0.8671875, 0.859375, 0.828125, 0.8515625, 0.8984375, 0.8984375, 0.8515625, 0.859375, 0.828125, 0.8671875, 0.84375, 0.8359375, 0.84375, 0.8046875, 0.859375, 0.8046875, 0.84375, 0.859375, 0.8828125, 0.8984375, 0.8984375, 0.8515625, 0.828125, 0.890625, 0.8828125, 0.953125, 0.890625, 0.859375, 0.8984375, 0.8671875, 0.84375, 0.890625, 0.890625, 0.859375, 0.921875, 0.84375, 0.890625, 0.8828125, 0.8828125, 0.859375, 0.8671875, 0.890625, 0.90625, 0.8828125, 0.9453125,
               0.8125, 0.8359375, 0.875, 0.8984375, 0.828125, 0.8984375, 0.8984375, 0.875, 0.8671875]
    batches = range(0, len(losslist))
    plt.plot(batches, acclist, color='r', label='acc')  # r表示红色
    plt.plot(batches, losslist, color='b', label='loss')  # 也可以用RGB值表示颜色

    #####非必须内容#########
    plt.xlabel('100-batch num')  # x轴表示
    plt.ylabel('y label')  # y轴表示
    plt.title("chart")  # 图标标题表示
    plt.legend()  # 每条折线的label显示
    #######################
    plt.savefig('loss_acc.jpg')  # 保存图片，路径名为test.jpg
    plt.show()  # 显示图片


def classification_report():
    from sklearn.metrics import classification_report
    y_true = [0, 1, 2, 2, 2]
    y_pred = [0, 0, 2, 2, 1]
    target_names = ['0', '1', '2', '3']
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)


def confusion_matrix():
    from sklearn.metrics import confusion_matrix
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt
    import numpy as np
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
              3,
              3, 3, 3, 3, 3, 3, 3, 3]
    y_pred = [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 3, 0, 3,
              0,
              3, 3, 3, 3, 3, 3, 3, 3]
    labels = [0, 1, 2, 3]
    maxtrix = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    print(maxtrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=maxtrix, display_labels=labels)
    disp.plot()
    plt.show()


def confusion_matrix2():
    from sklearn.metrics import confusion_matrix
    from matplotlib.font_manager import FontProperties
    import matplotlib.pyplot as plt
    import numpy as np
    y_true = []
    y_pred = []
    for i in range(102):
        y_true.append(0)
        y_pred.append(0)
    y_pred[0]=1
    y_pred[10]=1
    y_pred[11]=1
    y_pred[1] = 2
    y_pred[3] = 2
    y_pred[4] = 2
    y_pred[5] = 2
    y_pred[20]=10
    y_pred[21]=10
    y_pred[22]=10
    y_pred[23]=10
    y_pred[24]=10
    y_pred[25]=10
    y_pred[30]=3
    y_pred[40]=5
    y_pred[41]=5
    y_pred[50]=7
    y_pred[51]=8


    for i in range(103):
        y_true.append(1)
        y_pred.append(1)
    y_pred[110+0] = 0
    y_pred[110+1] = 0
    y_pred[110 + 2] = 0
    y_pred[110+3]=0
    y_pred[2+120] = 10
    y_pred[3+120] = 10
    y_pred[4+120] = 10
    y_pred[5+120] = 10
    y_pred[126] = 10
    y_pred[127] = 10
    y_pred[128] = 10
    y_pred[129] = 10
    y_pred[130]=10
    y_pred[140]=5
    y_pred[141]=5
    y_pred[150]=6
    y_pred[160]=9
    y_pred[161]=9

    for i in range(102):
        y_true.append(2)
        y_pred.append(2)
    y_pred[2 + 211] = 10
    y_pred[3 + 212] = 10
    y_pred[3 + 222] = 10
    y_pred[225]=10
    y_pred[230]=10
    y_pred[240]=0
    y_pred[241]=0
    y_pred[242]=0
    y_pred[250]=4
    y_pred[251]=4
    y_pred[252]=7
    y_pred[253]=1


    for i in range(103):
        y_true.append(3)
        y_pred.append(3)
    y_pred[2 + 301] = 4
    y_pred[3 + 301] = 10
    y_pred[3 + 301] = 2
    y_pred[3 + 311] = 2
    y_pred[350]=0
    y_pred[351] = 6
    y_pred[353]=6
    for i in range(5):
        y_pred[320+i]=10
        y_pred[330+i]=1

    for i in range(101):
        y_true.append(4)
        y_pred.append(4)
    y_pred[2 + 401] = 10
    y_pred[3 + 401] = 10
    y_pred[4 + 401] = 1
    y_pred[5 + 401] = 1
    for i in range(4):
        y_pred[420+i]=0
    y_pred[450]=6


    for i in range(101):
        y_true.append(5)
        y_pred.append(5)
    y_pred[2 + 510] = 10
    y_pred[3 + 511] = 10
    y_pred[3 + 512] = 10
    y_pred[520]=7
    y_pred[521]=7
    y_pred[522]=8

    for i in range(101):
        y_true.append(6)
        y_pred.append(6)
    y_pred[2 + 601] = 10
    y_pred[2 + 610] = 7
    y_pred[611]=8
    y_pred[612]=3
    y_pred[630]=5
    y_pred[641]=2
    y_pred[644]=1


    for i in range(104):
        y_true.append(7)
        y_pred.append(7)
    y_pred[2 + 701] = 10
    y_pred[2 + 701] = 3
    y_pred[3 + 701] = 2
    y_pred[10 + 701] = 2
    for i in range(5):
        y_pred[720+i]=10
    y_pred[730]=1
    y_pred[733]=1

    for i in range(103):
        y_true.append(8)
        y_pred.append(8)
    y_pred[2 + 801] = 10
    y_pred[11 + 801] = 10
    y_pred[12 + 801] = 10
    y_pred[810]=5
    y_pred[820]=4
    y_pred[830]=10
    y_pred[831]=10

    for i in range(107):
        y_true.append(9)
        y_pred.append(9)
    y_pred[2 + 901] = 10
    y_pred[10 + 901] = 10
    y_pred[11 + 901] = 10
    y_pred[12 + 901] = 10
    for i in range(5):
        y_pred[920+i]=8
    y_pred[930]=9
    y_pred[940]=3
    y_pred[941]=2
    y_pred[943]=2
    y_pred[944]=4

    for i in range(105):
        y_true.append(10)
        y_pred.append(10)
    for i in range(11):
        y_pred[i+1020]=0
    for i in range(9):
        y_pred[i+1040]=1
    for i in range(13):
        y_pred[i+1050]=2
    for i in range(6):
        y_pred[i+1080]=4
    for i in range(6):
        y_pred[i+1090]=8
    y_pred[i+1100]=9




    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    maxtrix = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    print(maxtrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=maxtrix, display_labels=labels)
    disp.plot()
    plt.show()


if __name__ == '__main__':
    #plt_loss_and_acc()

    #confusion_matrix2()

    print("Amazon Echo:",0.78125)
    print("Smart Things:", 0.55468)
    print("Tirby Speaker",0.77551)
    print("PIX-START Photo-frame:",0.64359)
    print("HP Printer:",0.83233)
    print("Netatmo Welcome:",0.61343)
    print("Withings Smart Baby Monitor:",0.59860)
    print("Samsung SmartCam:",0.78252)
    print("TP-Link Day Night Cloud camera:",0.49831)
    print("Dropcam:",0.86830)
    print("Insteon Camera:",0.79433)
    print("Belkin Wemo switch:",0.89729)
    print("TP-Link Smart plug:",0.45821)
    print("iHome:",0.49391)
    print("Belkin wemo motion sensor",0.75833)
    print("average acc:",0.69339)
    l=[1310,553,194,67,241,659,655,1261,310,4122,608,1092,39,50,1434]
    sum=0
    for i in range(len(l)):
        sum+=l[i]
    for i in range(len(l)):
        print('%.8f' % (l[i]/sum))

'''
关于评价指标：
https://blog.csdn.net/comway_Li/article/details/102758972


classification_report，在这个报告中：
y_true 为样本真实标签，y_pred 为样本预测标签；
support：当前行的类别在测试数据中的样本总量，如上表就是，在class 0 类别在测试集中总数量为1；
precision：精度=正确预测的个数(TP)/被预测正确的个数(TP+FP)；人话也就是模型预测的结果中有多少是预测正确的
recall:召回率=正确预测的个数(TP)/预测个数(TP+FN)；人话也就是某个类别测试集中的总量，有多少样本预测正确了；
f1-score:F1 = 2*精度*召回率/(精度+召回率)
micro avg：计算所有数据下的指标值，假设全部数据 5 个样本中有 3 个预测正确，所以 micro avg 为 3/5=0.6
macro avg：每个类别评估指标未加权的平均值，比如准确率的 macro avg，(0.50+0.00+1.00)/3=0.5
weighted avg：加权平均，就是测试集中样本量大的，我认为它更重要，给他设置的权重大点；比如第一个值的计算方法，(0.50*1 + 0.0*1 + 1.0*3)/5 = 0.70


confusion_matrix，混淆矩阵：
'''
