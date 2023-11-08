import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Head(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Head, self).__init__()
        self.conv5x5 = nn.Sequential(nn.Conv2d(input_channels, input_channels, 5, 1, 2, groups = input_channels, bias = False),
                                    #  nn.BatchNorm2d(input_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False)
                                    #  nn.BatchNorm2d(output_channels)
                                    ) 
        self.ca = ChannelAttention(input_channels)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return self.conv5x5(x)

class DetectHead(nn.Module):
    def __init__(self, input_channels, category_num, landmark_num):
        super(DetectHead, self).__init__()

        # self.conv1x1 = Conv1x1(input_channels, input_channels)
        self.obj_layers = Head(input_channels, 1)
        self.reg_layers = Head(input_channels, landmark_num*2)
        self.cls_layers = Head(input_channels, category_num)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
       
        obj = self.sigmoid(self.obj_layers(x))
        reg = self.reg_layers(x)
        cls = self.softmax(self.cls_layers(x))

        return torch.cat((obj, reg, cls), dim =1)