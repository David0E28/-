'''
made by XHU-WNCG
2022.4
'''
import cv2 as cv
import numpy as np
import os
import pandas as pd
from hparams import hparams
import matplotlib.pyplot as plt
import random

para = hparams()

img_path = para.img_path #导入原始图片地址

class For_Enhance_Dataset():

    def __init__(self, para):
        self.number, self.center_x_position, self.center_y_position, self.left_top_x_position, \
        self.left_top_y_position, self.right_bottom_x_position, self.right_bottom_x_position, \
        self.right_bottom_y_position, self.name = [], [], [], [], [], [], [], [], []

        self.origin_file_scp = para.origin_file_scp
        self.target_files = para.target_excel_path

        files = np.loadtxt(self.origin_file_scp, dtype='str')
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
m_Dataset = For_Enhance_Dataset(para)

class Slide_Data():
    def __init__(self, para):
        self.number, self.center_x_position, self.center_y_position, self.left_top_x_position, \
        self.left_top_y_position, self.right_bottom_x_position, self.right_bottom_x_position, \
        self.right_bottom_y_position, self.name = [], [], [], [], [], [], [], [], []

slide_data = Slide_Data(para)

window_y = 1000  #faster-rcnn input
window_x = 1000 #faster-rcnn input 大约1e6个像素
slide_x = 700
slide_y = 500

for idx in range(len(m_Dataset.number)):  ##根据为标注过的数据
    if m_Dataset.number[idx] != 0:
        img = cv.imread(m_Dataset.clean_files[idx])  # 读取数据
        a = int(m_Dataset.left_top_y_position[idx])
        b = int(m_Dataset.right_bottom_y_position[idx])
        c = int(m_Dataset.left_top_x_position[idx])
        d = int(m_Dataset.right_bottom_x_position[idx])

        n_X = 0
        while (window_x + n_X * slide_x) < len(img):
            n_Y = 0
            while (window_y + n_Y * slide_y) < len(img[0]):
                slide_window = img[n_X*slide_x:window_x + n_X*slide_x, n_Y*slide_y:window_y + n_Y*slide_y]         #滑动
                slide_data.name.append(str( '_%d_%d_' % (n_X, n_Y) + m_Dataset.name[idx]))                         #命名
                cv.imwrite(os.path.join(para.cut_data, slide_data.name[-1]), slide_window)
                if window_x + n_X * slide_x > c and  window_x + n_X * slide_x < d and window_y + n_Y * slide_y > b and\
                   window_y + n_Y * slide_y < a:
                    slide_data.number.append(m_Dataset.number)
                    slide_data.center_x_position.append(int(m_Dataset.center_x_position[idx]))
                    slide_data.center_y_position.append(int(m_Dataset.center_y_position[idx]))
                    slide_data.right_bottom_x_position.append(int(m_Dataset.right_bottom_x_position[idx]))
                    slide_data.right_bottom_y_position.append(int(m_Dataset.right_bottom_y_position[idx]))
                    slide_data.left_top_x_position.append(int(m_Dataset.left_top_x_position[idx]))
                    slide_data.left_top_y_position.append(int(m_Dataset.left_top_y_position[idx]))
                else:
                    slide_data.number.append(0)
                    slide_data.center_x_position.append('')
                    slide_data.center_y_position.append('')
                    slide_data.right_bottom_x_position.append('')
                    slide_data.right_bottom_y_position.append('')
                    slide_data.left_top_x_position.append('')
                    slide_data.left_top_y_position.append('')

                n_Y = n_Y + 1
            n_X = n_X + 1


dataframe = pd.DataFrame({'序号':range(len(slide_data.number)), '编号':slide_data.name, '类别':slide_data.number,
                          '中心x坐标': slide_data.center_x_position, '中心y坐标': slide_data.center_y_position,
                          "右上x坐标": slide_data.right_bottom_x_position, "右上y坐标": slide_data.right_bottom_y_position,
                          "左下x坐标": slide_data.left_top_x_position, "左下y坐标": slide_data.left_top_y_position})

#将DataFrame存储为csv,index表示是否显示行名，default=True， a模式使得循环读写不会覆盖上一次内容
dataframe.to_csv(para.cut_csv, index=False, sep=',', mode='a')

