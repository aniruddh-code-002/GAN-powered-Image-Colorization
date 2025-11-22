# discriminator.py
import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_filters=32):
        super().__init__()
        # input channels = grayscale(1) concat color(3) => 4
        f = base_filters
        layers = [
            nn.Conv2d(in_channels, f, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f, f*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(f*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f*2, f*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(f*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f*4, f*8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(f*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f*8, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, gray, color):
        # gray: (B,1,H,W) ; color: (B,3,H,W)
        x = torch.cat([gray, color], dim=1)
        return self.model(x)  # raw logits — no sigmoid  # [0,1] patch map

def build_discriminator(input_shape=(1,128,128), base_filters=32):
    return PatchDiscriminator(in_channels=input_shape[0]+3, base_filters=base_filters)
