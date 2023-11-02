import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Head, self).__init__()
        self.conv5x5 = nn.Sequential(nn.Conv2d(input_channels, input_channels, 5, 1, 2, groups = input_channels, bias = False),
                                     nn.BatchNorm2d(input_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False)
                                    #  nn.BatchNorm2d(output_channels)
                                    ) 
    
    def forward(self, x):
        return self.conv5x5(x)

class DetectHead(nn.Module):
    def __init__(self, input_channels, category_num, landmark_num):
        super(DetectHead, self).__init__()

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