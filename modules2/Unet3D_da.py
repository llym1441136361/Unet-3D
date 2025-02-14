
from modules_da import *


class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet3D, self).__init__()
        # features = [16, 48, 144, 432, 1296]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.inc = InConv(in_channels, 32, [9, 5, 1])
        self.down1 = Down(96, 48, [7, 3, 1])
        self.down2 = Down(144, 72, [5, 2, 1])
        self.down3 = Down(216, 108, [3, 2, 1])
        self.down4 = Down(324, 162, [2, 1, 1])

        self.up1 = Up(486, 324, 108, [3, 2, 1])
        self.up2 = Up(324, 216, 72, [5, 2, 1])
        self.up3 = Up(216, 144, 48, [7, 3, 1])
        self.up4 = Up(144, 96, 32, [9, 5, 1])
        self.outc = OutConv(96, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 80, 80, 64)
    net = UNet3D(3, 4)
    y = net(x)
    print("params: ", sum(p.numel() for p in net.parameters()))
    print(y.shape)
