import torch
import torch.nn as nn


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_ch, out_ch, dilation)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        # x = self.sigmoid(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=dilation[0], dilation=dilation[0]),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=dilation[0], dilation=dilation[0]),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=dilation[1], dilation=dilation[1]),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=dilation[1], dilation=dilation[1]),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=dilation[2], dilation=dilation[2]),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=dilation[2], dilation=dilation[2]),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),

        )
        # self.skip2 = nn.Sequential(
        #     nn.Conv3d(out_ch, out_ch, kernel_size=1, stride=1),
        #     nn.BatchNorm3d(out_ch),
        #     nn.ReLU(inplace=True),
        # )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return torch.cat([x1, x2, x3], dim=1)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dilation):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch, dilation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
