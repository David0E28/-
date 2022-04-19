'''
made by XHU-WNCG
2022.4
'''
class hparams():
    def __init__(self):
        #img
        self.small_img_path = "A:/BaiduNetdiskDownload/traindata/small_img"   #训练用压缩照片，压缩函数在datalider（commoned）
        self.object_img_path = "A:/BaiduNetdiskDownload/traindata/Object"     #裁剪后小目标的地址，可由Data——enhancement得到，不参与data_enhancement流程
        self.img_path = "A:/BaiduNetdiskDownload/traindata/data1"             #原照片，题给
        #scp
        self.train_file_scp = "scp/train.scp"                                 #题给照片地址scp格式
        self.origin_file_scp = "scp/data_enhancement.scp"                     #题给白板地址scp格式
        #csv
        self.target_excel_path = r"A:/BaiduNetdiskDownload/traindata/data2/target.csv"               #照片标注数据，题给
        self.EmpythBoard_excel_path = r"A:/BaiduNetdiskDownload/traindata/EmptyBoard/EmptyBoard.csv" #白板地址，题给
        self.enhanced_csv = r"A:/BaiduNetdiskDownload/traindata/data2/enhanced.csv"      #增强后标注数据

        # for enhancement data folder
        self.enhanced_data = 'A:/BaiduNetdiskDownload/traindata/enhanced_data'         #增强的照片放入这里

        self.N_epoch = 10
        self.dim_in = [256, 256]
        self.in_channals = 3
        self.dim_out = 0
        self.learning_rate = 1e-4
        self.batch_size = 12
        self.negative_slope = 1e-4
        self.dropout = 0.2
        self.layers = [2, 6, 9, 3]

