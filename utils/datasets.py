import os
import cv2
import numpy as np

import torch
import random

from utils.tool import compute_bounding_box

def random_crop(image, boxes, scale=0.75):
    height, width, _ = image.shape
    # random crop imgage
    cw, ch = random.randint(int(width * scale), width), random.randint(int(height * scale), height)
    cx, cy = random.randint(0, width - cw), random.randint(0, height - ch)

    roi = image[cy:cy + ch, cx:cx + cw]
    roi_h, roi_w, _ = roi.shape
    
    output = []
    for box in boxes:
        index, category = box[0], box[1]
        xc, yc = box[2] * width, box[3] * height
        x1, y1 = box[4] * width, box[5] * height
        x2, y2 = box[6] * width, box[7] * height
        x3, y3 = box[8] * width, box[9] * height
        x4, y4 = box[10] * width, box[11] * height
        
        xc, yc = (xc - cx)/roi_w, (yc - cy)/roi_h
        x1, y1 = (x1 - cx)/roi_w, (y1 - cy)/roi_h
        x2, y2 = (x2 - cx)/roi_w, (y2 - cy)/roi_h
        x3, y3 = (x3 - cx)/roi_w, (y3 - cy)/roi_h
        x4, y4 = (x4 - cx)/roi_w, (y4 - cy)/roi_h
        
        if x1 < 0 or x2 < 0 or x3 < 0 or x4 < 0:
            continue
        if y1 < 0 or y2 < 0 or y3 < 0 or y4 < 0:
            continue
        if x1 > 1 or x2 > 1 or x3 > 1 or x4 > 1:
            continue
        if y1 > 1 or y2 > 1 or y3 > 1 or y4 > 1:
            continue
        
        output.append([index, category, xc, yc, x1, y1, x2, y2, x3, y3, x4, y4])

    output = np.array(output, dtype=float)

    return roi, output

def random_narrow(image, boxes):
    height, width, _ = image.shape
    # random narrow
    cw, ch = random.randint(width, int(width * 1.25)), random.randint(height, int(height * 1.25))
    cx, cy = random.randint(0, cw - width), random.randint(0, ch - height)

    background = np.ones((ch, cw, 3), np.uint8) * 128
    background[cy:cy + height, cx:cx + width] = image

    output = []
    for box in boxes:
        index, category = box[0], box[1]
        xc, yc = box[2] * width, box[3] * height
        x1, y1 = box[4] * width, box[5] * height
        x2, y2 = box[6] * width, box[7] * height
        x3, y3 = box[8] * width, box[9] * height
        x4, y4 = box[10] * width, box[11] * height
        
        xc, yc = (xc + cx)/cw, (yc + cy)/ch
        x1, y1 = (x1 + cx)/cw, (y1 + cy)/ch
        x2, y2 = (x2 + cx)/cw, (y2 + cy)/ch
        x3, y3 = (x3 + cx)/cw, (y3 + cy)/ch
        x4, y4 = (x4 + cx)/cw, (y4 + cy)/ch
        
        if x1 < 0 or x2 < 0 or x3 < 0 or x4 < 0:
            continue
        if y1 < 0 or y2 < 0 or y3 < 0 or y4 < 0:
            continue
        if x1 > 1 or x2 > 1 or x3 > 1 or x4 > 1:
            continue
        if y1 > 1 or y2 > 1 or y3 > 1 or y4 > 1:
            continue

        output.append([index, category, xc, yc, x1, y1, x2, y2, x3, y3, x4, y4])

    output = np.array(output, dtype=float)

    return background, output

def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)

class TensorDataset():
    def __init__(self, path, img_width, img_height, aug=False):
        assert os.path.exists(path), "%s文件路径错误或不存在" % path

        self.aug = aug
        self.path = path
        self.data_list = []
        self.img_width = img_width
        self.img_height = img_height
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']

        # 数据检查
        with open(self.path, 'r') as f:
            for line in f.readlines():
                data_path = line.strip()
                if os.path.exists(data_path):
                    img_type = data_path.split(".")[-1]
                    if img_type not in self.img_formats:
                        raise Exception("img type error:%s" % img_type)
                    else:
                        self.data_list.append(data_path)
                else:
                    raise Exception("%s is not exist" % data_path)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        label_path = img_path.split(".")[0] + ".txt"

        # 加载图片
        img = cv2.imread(img_path)
        # 加载label文件
        if os.path.exists(label_path):
            label = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split(" ")
                    c = int(l[0])
                    if c > 17:
                        continue
                    
                    x = (float(l[1]) + float(l[3]) + float(l[5]) + float(l[7])) / 4
                    y = (float(l[2]) + float(l[4]) + float(l[6]) + float(l[8])) / 4
                    
                    # batch_idx, cls, xc, yc, x1, y1, x2, y2, x3, y3, x4, y4 
                    label.append([0, c, x, y, l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]])
            label = np.array(label, dtype=np.float32)

            # if label.shape[0]:
                # assert label.shape[1] == 6, '> 5 label columns: %s' % label_path
                #assert (label >= 0).all(), 'negative labels: %s'%label_path
                #assert (label[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s'%label_path
        else:
            raise Exception("%s is not exist" % label_path) 

        # 是否进行数据增强
        if self.aug:
            if random.randint(1, 10) % 2 == 0:
                img, label = random_narrow(img, label)
            else:
                img, label = random_crop(img, label, scale=random.randint(55, 100) / 100.0)

        img = cv2.resize(img, (self.img_width, self.img_height), interpolation = cv2.INTER_LINEAR) 

        # debug
        # for l in label:
        #     boxes = compute_bounding_box(torch.Tensor(l[2:]), n=5)
        #     for box in boxes:
        #         # print(box)
        #         bx, by, bw, bh = l[2], l[3], box[2], box[3]
        #         x1, y1 = int((bx - 0.5 * bw) * self.img_width), int((by - 0.5 * bh) * self.img_height)
        #         x2, y2 = int((bx + 0.5 * bw) * self.img_width), int((by + 0.5 * bh) * self.img_height)
        #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imwrite("debug.jpg", img)

        img = img.transpose(2,0,1)
        
        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    data = TensorDataset("/home/xuehao/Desktop/TMP/pytorch-yolo/widerface/train.txt")
    img, label = data.__getitem__(0)
    print(img.shape)
    print(label.shape)