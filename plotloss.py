'''
made by XHU-WNCG
2022.4
'''
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = 'SimHei'      # 显示中文
plt.rcParams['axes.unicode_minus'] = False      # 显示负号


def plotLoss(trainLoss, path, acc):
    # 训练集损失图
    trainLoss = np.array(trainLoss)
    # loss_history_train = tr_loss.detach().numpy()
    plt.plot(trainLoss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train loss')
    np.savetxt(path + '/train_loss.txt', trainLoss, delimiter=',')
    plt.savefig(path + '/train_loss.png')
    plt.close()

    # acc
    np.savetxt(path + 'acc.txt', acc, delimiter=',')
    acc = np.array(acc)
    plt.plot(acc)
    plt.xlabel('iters')
    plt.ylabel('percent')
    plt.title('acc')
    plt.savefig(path + '/acc.png')
    plt.close()