'''
made by XHU-WNCG
2022.4
'''
class hparams():
    def __init__(self):
        self.small_img_path = "A:/BaiduNetdiskDownload/traindata/small_img"
        self.img_path = "A:/BaiduNetdiskDownload/traindata/data1"
        self.file_scp = "scp/train.scp"
        self.target_excel_path = r"A:/BaiduNetdiskDownload/traindata/data2/target.csv"
        self.N_epoch = 10
        self.dim_in = [256, 256]
        self.in_channals = 3
        self.dim_out = 0
        self.learning_rate = 1e-4
        self.batch_size = 12
        self.negative_slope = 1e-4
        self.dropout = 0.2
        self.layers = [2, 6, 9, 3]

