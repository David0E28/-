'''
made by XHU-WNCG
2022.4
'''
import cv2 as cv
import numpy as np
import os
from hparams import hparams
import matplotlib.pyplot as plt
from dataset import WNCG_Dataset

idx = 4

para = hparams()

m_Dataset = WNCG_Dataset(para)

img_path = para.img_path #导入图片地址

file_name = []
for files in os.walk(img_path):
        for file in files[2]:
            fileName = os.path.join(img_path, file)
            if (fileName.endswith(".jpg")):file_name.append(fileName)


if(m_Dataset.number[idx] != 0):
    img = cv.imread(file_name[idx])
    a = int(m_Dataset.left_top_y_position[idx]),
    b = int(m_Dataset.right_bottom_y_position[idx])
    c = int(m_Dataset.left_top_x_position[idx]),
    d = int(m_Dataset.right_bottom_x_position[idx])


    img = (img[a[0]:b])
    print(img)

    plt.imshow(img)
    plt.show()