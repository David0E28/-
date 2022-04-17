'''
made by XHU-WNCG
2022.4
'''
import torch
from hparams import hparams
import os
import numpy as np
import pandas as pd
import cv2

# 定义device
device = torch.device("cuda:0")

#该函数从GPU中拿到模型, 从内存里拿test数据，进行预测
def eval(model, test_X, test_Y):

    false = 0.

    for idx in range(len(test_Y)):
        test_X[idx] = test_X[idx].to(device)
        test_Y[idx] = test_Y[idx].to(device)
        model.eval()
        with torch.no_grad():
            X_out = model(x=test_X[idx])

            for i in range(len(test_Y[idx])):
                if(test_Y[idx][i] != 0 and X_out[i][0] <= X_out[i][1]):false = false + 1
    print('网络的准确率为:', (1 - false / (len(test_Y[0]) * len(test_Y))))
    return 1 - false / (len(test_Y[0]) * len(test_Y))







    # para = hparams()
    #
    # files = np.loadtxt(para.file_scp_test, dtype='str')
    #
    # clean_files = files[:].tolist()
    #
    # target_csv_files = pd.read_csv(para.target_test_path, encoding='utf-8')
    #
    # target_csv_files = target_csv_files.values
    #
    # number, res = [], []
    #
    # for _ in target_csv_files:
    #     number.append(_[2])  ##得到lable
    #
    # false = 0.
    # # 读取训练好的模型
    #
    # for _ in clean_files:
    #     res.append(eval_file(_, model))
    #
    # for i in range(len(number)):
    #     if(number[i] != 0 and res[i][0][0] <= res[i][0][1]):
    #         false = false + 1
    # print('网络的准确率为:', (1 - false / len(number)))
    # return 1 - (false / len(number))

    

                
                
                
                
               
    
 
    
    