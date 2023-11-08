from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .aff import AFF

def BasicConv(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


def Conv(filter_in, filter_out, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))



class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, 2, 2)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x



class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]

        out = self.conv(fused_out_reduced)

        return out


class SAF12(nn.Module):
    '''
    Simple Attention Fusion
    '''
    
    def __init__(self, input_channels=[64, 128]):
        super(SAF12, self).__init__()

        self.downsample = Downsample_x2(input_channels[0], input_channels[1])
        
        self.aff = AFF(channels=input_channels[1], r=4)

    def forward(self, x1, x2):
        x1 = self.downsample(x1)
        return self.aff(x1, x2)

class SAF23(nn.Module):
    '''
    Simple Attention Fusion
    '''
    
    def __init__(self, input_channels=[64, 128]):
        super(SAF23, self).__init__()

        self.upsample = Upsample(input_channels[1], input_channels[0])
        self.aff = AFF(input_channels[0])

    def forward(self, x1, x2):
        x2 = self.upsample(x2)
        return self.aff(x1, x2)

class SAF13(nn.Module):
    '''
    Simple Attention Fusion
    '''
    
    def __init__(self, input_channels=[40, 160]):
        super(SAF13, self).__init__()
        inter_channel = input_channels[1] // 2
        self.upsample = Upsample(input_channels[1], inter_channel)
        self.downsample = Downsample_x2(input_channels[0], inter_channel)
        self.aff = AFF(inter_channel)

    def forward(self, x1, x2):
        x1 = self.downsample(x1)
        x2 = self.upsample(x2)
        return self.aff(x1, x2)


if __name__ == '__main__':
    modle = SAF12(input_channels=[40, 80])
    x1 = torch.randn(128, 40, 52, 52)
    x2 = torch.randn(128, 80, 26, 26)
    y = modle(x1, x2)
    print(y.shape)


        

