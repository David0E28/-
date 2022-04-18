'''
made by XHU-WNCG
2022.4
'''
import torch
import torch.nn as nn
from hparams import hparams
from torch.utils.data import Dataset,DataLoader
from dataset import WNCG_Dataset
from model_FPN import FPN, DNN
from loss_F import categorical_crossentropy
import os
import time
import torch.optim as optim
from plotloss import plotLoss
import eval
if __name__ == "__main__":
    acc = []
    
    # 定义device
    device = torch.device("cuda:0")
    
    # 获取模型参数
    para = hparams()
    
    # 定义模型
    m_model = DNN(para.layers)
    m_model = m_model.to(device)
    m_model.train()
    #233
    # 定义损失函数
    #loss_fun = categorical_crossentropy()
    loss_fun = nn.CrossEntropyLoss()
    loss_fun = loss_fun.to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(
        params=m_model.parameters(),
        lr=para.learning_rate)
    
    # 定义数据集
    m_Dataset= WNCG_Dataset(para)
    m_DataLoader = DataLoader(m_Dataset, batch_size=para.batch_size, shuffle=True, num_workers=1)
    
    # 定义训练的轮次 
    n_epoch = para.N_epoch
    n_step = 0
    loss_total = 0
    loss_list = []
    test_X, test_Y = [], [] ##因为test文件在epoch后校验，所以要存在内存中，和tarin文件逻辑不同
    save_file = os.path.join('save', time.strftime('%m-%d-%H-%M' + '/'))
    os.makedirs(save_file, exist_ok= True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.3, patience=2, verbose=True)
    for epoch in range(n_epoch):
        # 遍历dataset中的数据
        for i_batch, sample_batch in enumerate(m_DataLoader):
            if(i_batch % 10 == 0):                                  #随机取出十分之一个batch作为验证集,依托于Dataloder_shuffle的伪k折验证法
                test_X.append(sample_batch[0])
                test_Y.append(sample_batch[1])
            train_X = sample_batch[0]
            train_Y = sample_batch[1]
            train_X = train_X.to(device)
            train_Y = train_Y.to(device)

            m_model.zero_grad()
            # 得到网络输出
            output = m_model(x=train_X)
            output = output.squeeze()    ##tensor([batchsize,class])传不入
            train_Y = train_Y.squeeze()  ##改为tensor（[batchsize])


            # 计算损失函数
            loss = loss_fun.forward(output, train_Y.long())
            loss.requires_grad_(True)
            # 误差反向传播
            # optimizer.zero_grad()
            loss.backward()

            # 进行参数更新
            # optimizer.zero_grad()
            optimizer.step()
            
            n_step = n_step+1
            loss_total = loss_total+loss
            
            # 每100 step 输出一次中间结果
            if n_step %10 == 0:
                print("epoch = %02d  step = %04d  loss = %.4f"%(epoch,n_step, loss))
                print(time.strftime('%m-%d-%H-%M-%S'))
        
        # 训练结束一个epoch 计算一次平均结果
        loss_mean = loss_total/n_step
        loss_list.append(loss_mean)
        print("epoch = %02d mean_loss = %f"%(epoch,loss_mean))
        loss_total = 0
        n_step = 0
        scheduler.step(loss)

        ##输出准确率
        acc.append(eval.eval(m_model, test_X, test_Y))

        # 进行模型保存
        save_name = os.path.join(save_file, 'model_%d_%.4f.pth' % (epoch, loss_mean))
        torch.save(m_model, save_name)

    #绘制train_loss曲线
    plotLoss(loss_list, save_file, acc)












