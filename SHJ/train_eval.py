# coding: UTF-8
import itertools
import torch
from sklearn import metrics
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def plt_loss_and_acc(total_batch:int,losslist,acclist):
    batches=range(0,total_batch)
    plt.plot(batches, acclist, color='r', label='acc')  # r表示红色
    plt.plot(batches, losslist, color='b', label='loss')  # 也可以用RGB值表示颜色

    plt.xlabel('batch num')  # x轴表示
    plt.ylabel('y label')  # y轴表示
    plt.title("loss acc曲线")  # 图标标题表示
    plt.legend()  # 每条折线的label显示
    #######################
    plt.savefig('loss_acc.jpg')  # 保存图片，路径名为test.jpg
    plt.show()  # 显示图片

def train(train_iter, valid_iter, config, model):
    '''

    :param train_iter: 训练集迭代器
    :param valid_iter: 验证集迭代器
    :param config: 模型配置
    :param model: 模型
    :return: 无
    '''
    # 超参数
    use_gpu = torch.cuda.is_available()  # torch可以使用GPU时会返回true
    epoch = config.epoch

    optimizer= torch.optim.Adam(model.parameters(), lr=config.learn_rate, betas=(0.9, 0.999),weight_decay=0.01)  # 后两项时默认值
    if use_gpu:
        model = model.cuda()  # 把网络模型放到gpu上

    model.train()  # 训练一定要调用.train()，作用是启用batch normalization和drop out。测试的时候再调用.eval()

    total_batch = 0  # 记录进行到多少batch,总的batch数目
    dev_best_loss = float('inf')  # 表示正无穷
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    lossList=[]#作损失曲线图用
    accList=[]#作acc曲线图用
    index=0#作图用

    for e in range(epoch):  # 每一次迭代都是针对全体数据做的,但是参数更新一batch一次
        for _, (trains, labels) in enumerate(train_iter):  # 批数据的下标，该批数据的data及对应的label
            if use_gpu:
                trains, labels = trains.cuda(), labels.cuda()  # 把数据放到GPU上
            else:
                pass
            outputs = model(trains)  # 送进去的训练数据的第一个维度是一batch_size大小,即每次训练的样本数目是一batch
            loss = F.cross_entropy(outputs, labels)  # 计算损失
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新网络参数
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()  # 对每个样本输出是一个softmax向量，最大值的下标就是预测类别
                train_acc = metrics.accuracy_score(true, predic)#输出本次训练的准确率
                dev_acc, dev_loss = evaluate(config, model, valid_iter)  # 在验证集上评估
                accList.append(dev_acc)
                lossList.append(dev_loss.item())
                index+=1
                if dev_loss <= dev_best_loss:  # 验证集上loss变小了
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)  # model.state_dict () 是浅拷贝，返回的参数仍然会随着网络的训练而变化
                    improve = '****'
                    last_improve = total_batch#记录最近一次loss下降的batch数
                else:  # 验证集上没有变好
                    improve = ''
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, {5}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > 1000:
                # 验证集loss超过1000batch仍没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    print("train over ,has save model.ckpt----------")
    plt_loss_and_acc(index, losslist=lossList, acclist=accList)#绘制acc-loss曲线



def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))#加载模型参数

    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)#测试集评估
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)

def evaluate(config, model, data_iter, test=False):
    '''
    :param config:模型配置
    :param model:模型
    :param data_iter：验证集迭代器或者测试集迭代器
    :return:对验证集，返回：全体样本的准确率、平均batch loss；对测试集，返回：全体样本的准确率、平均batch loss、sklearn的评价指标报告、混淆矩阵
    '''
    model.eval()#测试的时候不要DropOut
    loss_total = 0#整个数据集data上全体样本的损失之和
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            if torch.cuda.is_available():
                texts, labels = texts.cuda(), labels.cuda()  # 把数据放到GPU上
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        plot_confusion_matrix(confusion,config.class_list,normalize=True)#绘制混淆矩阵
        plt.show()
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)  # 保留两位有效数字
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap,aspect ='auto')#将数组的值以图片的形式展示出来,数组的值对应着不同的颜色深浅,而数值的横纵坐标就是数组的索引
    plt.colorbar()
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')