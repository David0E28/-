'''
made by XHU-WNCG
2022.4
'''
import torch
import torch.nn as nn
from hparams import hparams
import torch.nn.functional as F


para = hparams()
# ResNet的基本Bottleneck类
class Bottleneck(nn.Module):
    expansion = 4  # 通道倍增数

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(planes, self.expansion * planes, 1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.LeakyReLU(out)
        return out


# FNP的类，初始化需要一个list，代表RESNET的每一个阶段的Bottleneck的数量
class FPN(nn.Module):
    def __init__(self, layers):
        super(FPN, self).__init__()
        self.inplanes = 64
        # 处理输入的C1模块（C1代表了RestNet的前几个卷积与池化层）
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # 搭建自下而上的C2，C3，C4，C5
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        # 对C5减少通道数，得到P5
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)
        # 3x3卷积融合特征
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        # 横向连接，保证通道数相同
        self.latlayer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, 256, 1, 1, 0)

        self.SIZE = [64, 32, 16, 8]


        #FC层
        self.reduce_ = nn.Conv2d(256, 1, 1, 1, 1)
        self.lin1 = nn.Linear(in_features=self.SIZE[1]*self.SIZE[1]*256, out_features=256)  ###
        self.lin2 = nn.Linear(in_features=self.SIZE[3]*self.SIZE[3]*256, out_features=256)  ###
        self.lin3 = nn.Linear(in_features=512, out_features=2)



    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != Bottleneck.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, Bottleneck.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * planes)
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    # 自上而下的采样模块
    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # 自下而上
        c1 = self.maxpool(self.LeakyReLU(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # 自上而下
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        #p2 = self._upsample_add(p3, self.latlayer3(c2))
        # 卷积的融合，平滑处理
        #p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        #p2 = self.smooth3(p2)

        self.PA = p3
        self.PB = p5

        x_1 = self.reduce_(self.PA)
        x_2 = self.reduce_(self.PB)
        x_1 = self.PA.view(-1,self.SIZE[1]*self.SIZE[1]*256) ###
        x_2 = self.PB.view(-1,self.SIZE[3]*self.SIZE[3]*256) ###
        x_1 = self.lin1(x_1)
        x_2 = self.lin2(x_2)
        x = torch.cat((x_1, x_2), 1)
        out = (self.lin3(x))

        return out


class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.BN1 = nn.BatchNorm2d(3)  #(N, C, H, W)
        self.lin1 = nn.Linear(in_features=256, out_features=128) ##in_feature = W
        self.lin2 = nn.Linear(in_features=128, out_features=1) ##
        self.lin3 = nn.Linear(in_features=para.in_channals * 256 * 1, out_features=32)  ##
        self.lin4 = nn.Linear(in_features=32, out_features=2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):   #(N, C, H, W)
        x = self.BN1(x)
        x = (F.relu(self.lin1(x)))
        x = self.dropout(x)
        x = (F.relu(self.lin2(x)))
        x = x.view(-1, para.in_channals * 256 * 1)  ##
        x = self.lin3(x)
        x = F.relu(self.lin4(x))

        return x



        
if __name__ == "__main__":
    para = hparams()
    m_model = FPN([2,6,9,3])
    print(m_model)
    x = torch.randn(para.batch_size, para.in_channals, para.dim_in[0], para.dim_in[1])
    print(x)
    y = m_model(x)
    print(y)

        
        