import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import cv2
import os
import random
import numpy as np

def random_crop(image, boxes):
    height, width, _ = image.shape
    # random crop imgage
    cw, ch = random.randint(int(width * 0.75), width), random.randint(int(height * 0.75), height)
    cx, cy = random.randint(0, width - cw), random.randint(0, height - ch)

    roi = image[cy:cy + ch, cx:cx + cw]
    roi_h, roi_w, _ = roi.shape
    
    output = []
    for box in boxes:
        index, category = box[0], box[1]
        bx, by = box[2] * width, box[3] * height
        bw, bh = box[4] * width, box[5] * height

        bx, by = (bx - cx)/roi_w, (by - cy)/roi_h
        bw, bh = bw/roi_w, bh/roi_h

        output.append([index, category, bx, by, bw, bh])

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
        bx, by = box[2] * width, box[3] * height
        bw, bh = box[4] * width, box[5] * height

        bx, by = (bx + cx)/cw, (by + cy)/ch
        bw, bh = bw/cw, bh/ch

        output.append([index, category, bx, by, bw, bh])

    output = np.array(output, dtype=float)

    return background, output

def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, path, img_width, img_height, aug=False, split='train2017'):
        self.root_dir = path
        annotation_file = os.path.join(path, "annotations", "instances_" + split + ".json")

        self.split = split
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.categories = self.coco.loadCats(self.coco.getCatIds())

        self.stuff_to_obj = {}
        for i, category in enumerate(self.categories):
            self.stuff_to_obj[category['id']] = i

        # self.transform = transform
        # print(self.categories)
        self.aug = aug

        self.input_width = img_width
        self.input_height = img_height

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, self.split,  img_info['file_name'])
        # print(img_path)
        img = cv2.imread(img_path)

        # img = Image.open(img_path).convert("RGB")
        # if self.transform is not None:
            # img = self.transform(img)

        # Convert annotations to YOLO format
        label = []
        for obj in target:
            x, y, w, h = obj['bbox']
            x_center = x + w / 2
            y_center = y + h / 2
            image_width = img_info['width']
            image_height = img_info['height']
            
            stuff_id = obj['category_id']
            if stuff_id not in self.stuff_to_obj:
                continue
            
            cls = self.stuff_to_obj[stuff_id]

            assert cls >= 0 and cls <= 79, "cls error {}".format(cls)

            label.append([0, cls, x_center / image_width, y_center / image_height, w / image_width, h / image_height])
        label = np.array(label, dtype=np.float32)

        if self.aug:
            if random.randint(1, 10) % 2 == 0:
                img, label = random_narrow(img, label)
            else:
                img, label = random_crop(img, label)

        img = cv2.resize(img, (self.input_width, self.input_height), interpolation = cv2.INTER_LINEAR)
        img = img.transpose(2,0,1)

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    # Example usage
    root_dir = "/mnt/zcf/archivecc/coco2017"
    annotation_file = "/mnt/zcf/archivecc/coco2017/annotations/instances_train2017.json"

    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = COCODataset(root_dir, 416, 416, aug=True, split='val2017')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Iterate over the dataloader
    for images, targets in dataloader:
        # Do something with the images and targets
        # print(images.shape, targets.shape)
        # print(targets)
        # break
        continue