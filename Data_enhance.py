'''
made by XHU-WNCG
2022.4
'''
import cv2 as cv
import numpy as np
import os
import  pandas as pd
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

class newData():
    def __init__(self, para):
        self.number, self.center_x_position, self.center_y_position, self.left_top_x_position, \
        self.left_top_y_position, self.right_bottom_x_position, self.right_bottom_x_position, \
        self.right_bottom_y_position, self.name = [], [], [], [], [], [], [], [], []

        self.EmpythBoard_excel_path = para.EmpythBoard_excel_path
        files = np.loadtxt(self.EmpythBoard_excel_path, dtype='str')
        self.EmptyBoard = files[:].tolist()
        print("空白模板量：", len(self.EmptyBoard))
newData = newData(para)


"""
lacked_object :the lacked object you want to enhance ,but you can only enhance one object at a time .
number : how many time you want spand object . number = ori_number * number

retrun : a newDate Type object
"""
def Copy_and_paste(lacked_object, number):
    def seamlessClone(out_path, obj,im, center, flag):
    # Create an all white mask
        mask = 255 * np.zeros(obj.shape, obj.dtype)

    # Seamlessly clone src into dst and put the results in output
        if flag == 0:
            mode = cv.NORMAL_CLONE
        elif flag == 1:
            mode = cv.MIXED_CLONE
        else:
            mode = cv.MONOCHROME_TRANSFER
        clone = cv.seamlessClone(obj, im, mask, center, mode)

    # Write results
        cv.imwrite(out_path, clone)

    if lacked_object == 0 :
        print("编号0为白板呦")
        return
    for _ in range(number):
            for idx in range(len(m_Dataset.number)):  ##根据为原标注过数据的

                Pick_a_board = random.randint(1, len(newData.EmptyBoard))

                if(m_Dataset.number[idx] == lacked_object):
                    img = cv.imread(m_Dataset.clean_files[idx])  # 读取数据
                    a = int(m_Dataset.left_top_y_position[idx])
                    b = int(m_Dataset.right_bottom_y_position[idx])
                    c = int(m_Dataset.left_top_x_position[idx])
                    d = int(m_Dataset.right_bottom_x_position[idx])


                    object = (img[a: b, c: d])

                    #生成合理的白板
                    New_Board = cv.imread(os.path.join(para.img_path, newData.EmptyBoard[Pick_a_board]))  # 读取白板

                    #生成合理的位移
                    random_x = random.randint(-1000, 1000)
                    random_y = random.randint(-1000, 1000)




                    #根据位移合成
                    newData.name.append(str(m_Dataset.number[idx])+'_'+newData.EmptyBoard[Pick_a_board])

                    newData.number.append(m_Dataset.number[idx])

                    newData.center_x_position.append(int(m_Dataset.center_x_position[idx]) + random_x)
                    newData.center_y_position.append(int(m_Dataset.center_y_position[idx]) + random_y)
                    newData.right_bottom_x_position.append(int(m_Dataset.right_bottom_x_position[idx]) + random_x)
                    newData.right_bottom_y_position.append(int(m_Dataset.right_bottom_y_position[idx]) + random_y)
                    newData.left_top_x_position.append(int(m_Dataset.left_top_x_position[idx]) + random_x)
                    newData.left_top_y_position.append(int(m_Dataset.left_top_y_position[idx]) + random_y)

                    seamlessClone(out_path=os.path.join(para.enhanced_data, str(m_Dataset.number[idx])+'_'+newData.EmptyBoard[Pick_a_board]),
                                  obj=object,
                                  im=New_Board,
                                  center=(int(m_Dataset.center_x_position[idx]) + random_x, int(m_Dataset.center_y_position[idx]) + random_y),
                                  flag=4)




                    plt.imshow(New_Board)
                    plt.show()


Copy_and_paste(398, 1)



dataframe = pd.DataFrame({'序号':range(len(newData.number)),'编号':newData.name,'类别':newData.number,
                          '中心x坐标': newData.center_x_position,'中心y坐标': newData.center_y_position,
                          "右上x坐标": newData.right_bottom_x_position,"右上y坐标": newData.right_bottom_y_position,
                          "左下x坐标": newData.left_top_x_position,"左下y坐标": newData.left_top_y_position})

#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv(para.enhanced_csv, index=False, sep=',',encoding='utf-8')


