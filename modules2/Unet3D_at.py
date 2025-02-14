import torch
import torch.nn as nn
from modules_at import *


class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet3D, self).__init__()
        features = [32, 64, 128, 256]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.inc = InConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.se1 = SE_Block(features[1])
        self.down2 = Down(features[1], features[2])
        self.se2 = SE_Block(features[2])
        self.down3 = Down(features[2], features[3])
        self.se3 = SE_Block(features[3])
        self.down4 = Down(features[3], features[3])
        self.se4 = SE_Block(features[3])

        self.up1 = Up(features[3], features[3], features[2])
        self.se5 = SE_Block(features[2])
        self.up2 = Up(features[2], features[2], features[1])
        self.se6 = SE_Block(features[1])
        self.up3 = Up(features[1], features[1], features[0])
        self.se7 = SE_Block(features[0])
        self.up4 = Up(features[0], features[0], features[0])
        self.se8 = SE_Block(features[0])
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.se1(x2)
        x3 = self.down2(x2)
        x3 = self.se2(x3)
        x4 = self.down3(x3)
        x4 = self.se3(x4)
        x5 = self.down4(x4)
        x5 = self.se4(x5)

        x = self.up1(x5, x4)
        x = self.se5(x)
        x = self.up2(x, x3)
        x = self.se6(x)
        x = self.up3(x, x2)
        x = self.se7(x)
        x = self.up4(x, x1)
        x = self.se8(x)
        x = self.outc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 160, 160, 128)
    net = UNet3D(3, 4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)
