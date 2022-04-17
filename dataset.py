'''
made by XHU-WNCG
2022.4
'''
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from hparams import hparams
import random
import pandas as pd
import cv2

# 获取模型参数
para = hparams()
    
    

class WNCG_Dataset(Dataset):
    
    def __init__(self, para):
        self.number, self.center_x_position, self.center_y_position, self.left_top_x_position,\
        self.left_top_y_position, self.right_bottom_x_position, self.right_bottom_x_position, \
        self.right_bottom_y_position,self.name = [], [], [], [], [], [], [], [], []

        self.file_scp = para.file_scp
        self.target_files = para.target_excel_path

        files = np.loadtxt(self.file_scp,dtype = 'str')
        self.clean_files = files[:].tolist()

        self.target_csv_files = pd.read_csv(self.target_files, encoding='ANSI')
        self.target_csv_files = self.target_csv_files.values
        for _ in self.target_csv_files:
            self.name.append(_[1])
            self.number.append(_[2])
            self.center_x_position.append(_[4])
            self.center_y_position.append(_[5])
            self.left_top_x_position.append(_[6])
            self.left_top_y_position.append(_[7])
            self.right_bottom_x_position.append(_[8])
            self.right_bottom_y_position.append(_[9])

        print("训练数据量：", len(self.clean_files))
    
    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self,idx):
        # 读取待分类图像,第一次跑需要生成小图片数据

        img_cv = cv2.imread(self.clean_files[idx])  # 读取数据
        img_cv = img_cv.astype('float32')
        img_cv = cv2.resize(src=img_cv,  dsize=para.dim_in)
        #cv2.imwrite(os.path.join(para.small_img_path, self.name[idx]), img_cv)
        img_cv = np.transpose(img_cv, (2, 0, 1))



        #  读取标签文件
        target = self.number[idx]
        target = np.array(target, dtype=np.float32)
        target = np.expand_dims(target, 0)  ##扩充dim，否则无法形成batch（ tensor（1）-》tensor（[1]） ）

        #转化为独热码
        #One_hot = np.zeros(29)
        dict = {
            0: 0,
            6: 1,
            7: 2,
            8: 3,
            9: 4,
            10: 5,
            25: 6,
            41: 7,
            105: 8,
            110: 9,
            115: 10,
            148: 11,
            156: 12,
            222: 13,
            228: 14,
            235: 15,
            256: 16,
            280: 17,
            310: 18,
            387: 19,
            392: 20,
            394: 21,
            398: 22,
            401: 23,
            402: 24,
            430: 25,
            480: 26,
            485: 27,
            673: 28,
        }
        #One_hot[dict.get(target[0])] = 1

        Two_hot = np.zeros(1)
        if(target[0] != 0):
            Two_hot = 1
            Two_hot = np.expand_dims(Two_hot, 0)



        # 转为torch格式
        X_train = torch.from_numpy(img_cv)
        #Y_target = torch.Tensor(Two_hot)
        Y_target = torch.from_numpy(Two_hot)

        return X_train.float(), Y_target.float()

# def my_collect(batch):
#     batch_X = [item[0] for item in batch]
#     batch_Y = [item[1] for item in batch]
#
#
#     batch_X = torch.cat(batch_X, -1)
#     batch_Y = torch.cat(batch_Y, -1)    ##根据CrossEntropyLoss()的源码，输出维度为（batchsize）
#
#
#     return[batch_X.float(),batch_Y.float()]
    
    
if __name__ == '__main__':
    
    # 数据加载测试
    para = hparams()
    
    m_Dataset= WNCG_Dataset(para)
    
    #m_DataLoader = DataLoader(m_Dataset, batch_size=2, shuffle=True, num_workers=1)
    m_DataLoader = DataLoader(m_Dataset, batch_size=2, shuffle=True, num_workers=1,)
    
    for i_batch, sample_batch in enumerate(m_DataLoader):
        train_X = sample_batch[0]
        train_Y = sample_batch[1]
        print(train_X.shape)
        #print(train_Y.shape)