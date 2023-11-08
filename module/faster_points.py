import torch
import torch.nn as nn

from .fasternet import fastenet_simplest
from .shufflenet import ShuffleNetV2
from .head import DetectHead
from .afpn import AFPN
from .spp import SPP
from .saf import SAF12, SAF23, SAF13
# from .coordconv import AddCoords

class Detector(nn.Module):
    def __init__(self, category_num, keypoints_num, load_param):
        super(Detector, self).__init__()
        
        self.stride = [8, 16, 32]

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 40, 80, 160]
        # self.stage_out_channels = [-1, 24, 80, 160, 320]
        # self.stage_out_channels = [-1, 24, 48, 96, 192]
        # self.backbone = ShuffleNetV2(self.stage_repeats, self.stage_out_channels, load_param)
        self.backbone = fastenet_simplest()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.SPP = SPP(sum(self.stage_out_channels[-3:]), self.stage_out_channels[-2])
         
        self.detect_head = DetectHead(self.stage_out_channels[-2], category_num, keypoints_num)
        # self.afpn = AFPN(in_channels=self.stage_out_channels[-3:], out_channels=self.stage_out_channels[-3:])
        
        # self.add_coords = AddCoords(with_r=False)
        # self.saf12 = SAF12(input_channels=[40, 80])
        # self.saf23 = SAF23(input_channels=[80, 160])
        # self.saf13 = SAF13(input_channels=[40, 160])

    def forward(self, x):
        # x = self.add_coords(x)
        # print(x.shape)
        P1, P2, P3 = self.backbone(x)
        # P1, P2, P3 = self.afpn([P1, P2, P3])

        # print(P1.shape, P2.shape, P3.shape)

        # F1 = self.saf12(P1, P2)
        # F2 = self.saf23(P2, P3)
        # F3 = self.saf13(P1, P3)

        P1 = self.pool(P1)
        P3 = self.upsample(P3)
        # P = torch.cat((F1, F2, F3), dim=1)
       
        P = torch.cat((P1, P2, P3), dim=1)
        # print(P.shape)

        
        # y = self.conv1x1(P)
        y = self.SPP(P)

        # print(y.shape)
        return self.detect_head(y) 
    
if __name__ == "__main__":
    model = Detector(80, False)
    test_data = torch.rand(1, 3, 352, 352)
    torch.onnx.export(model,                    #model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     "./test.onnx",             # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=11,          # the ONNX version to export the model to
                     do_constant_folding=True)  # whether to execute constant folding for optimization

