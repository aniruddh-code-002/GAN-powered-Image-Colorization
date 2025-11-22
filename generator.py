import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        # After concatenation, channel count doubles, so in_ch for DoubleConv = out_ch * 2
        self.conv = DoubleConv(out_ch * 2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad to match size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY or diffX:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_filters=32):
        super().__init__()
        f = base_filters
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.bottleneck = DoubleConv(f * 8, f * 16)

        self.up3 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up1 = Up(f * 4, f * 2)
        self.final_up = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()  # output in [0,1]

    def forward(self, x):
        c1 = self.inc(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        b = self.bottleneck(c4)

        u3 = self.up3(b, c4)
        u2 = self.up2(u3, c3)
        u1 = self.up1(u2, c2)
        u0 = self.final_up(u1)

        out = self.outc(u0)
        return self.out_act(out)


def build_generator(input_shape=(1, 128, 128), base_filters=32):
    return UNetGenerator(in_channels=input_shape[0], out_channels=3, base_filters=base_filters)
