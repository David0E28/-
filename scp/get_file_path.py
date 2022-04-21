'''
made by XHU-WNCG
2022.4
'''
import os
import pandas as pd
# 将用于训练的语音数据集地址写入到文本文件中
# basePath的路径要采用用绝对路径!!!!!!!!!!!!!!!!!!!!!!!!!!!
trainCleanSpeechRootDir = r'A:/BaiduNetdiskDownload/traindata/cut_data'
if __name__ == '__main__':

        with open('./train.scp', 'w', encoding='utf-8') as fp:
                name = []
                target_csv_files = pd.read_csv(r"A:/BaiduNetdiskDownload/traindata/data2/cut.csv", encoding='ANSI')
                target_csv_files = target_csv_files.values
                for _ in target_csv_files:
                        fileName = os.path.join(trainCleanSpeechRootDir, _[1])
                        print(fileName)
                        fp.write('%s\n' % fileName)


        # with open('./text.scp', 'w', encoding='utf-8') as fp:
        #         name = []
        #         target_csv_files = pd.read_csv(r"A:/BaiduNetdiskDownload/traindata/data2/text.csv", encoding='utf-8')
        #         target_csv_files = target_csv_files.values
        #         for _ in target_csv_files:
        #                 print(_[1])
        #                 fileName = os.path.join(trainCleanSpeechRootDir, _[1])
        #                 print(fileName)
        #                 fp.write('%s\n' % fileName)