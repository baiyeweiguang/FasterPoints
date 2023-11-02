import torch
import numpy as np
from tqdm import tqdm
from utils.tool import *
import cv2

from cocotools.coco import COCO
from cocotools.cocoeval import COCOeval

class CocoDetectionEvaluator():
    def __init__(self, names, device, num_keypoints):
        self.device = device
        self.num_keypoints = num_keypoints
        self.classes = []
        with open(names, 'r') as f:
            for line in f.readlines():
                self.classes.append(line.strip())
    
    def coco_evaluate(self, gts, preds):
        # Create Ground Truth
        coco_gt = COCO()
        coco_gt.dataset = {}
        coco_gt.dataset["images"] = []
        coco_gt.dataset["annotations"] = []
        k = 0
        for i, gt in enumerate(gts):
            for j in range(gt.shape[0]):
                k += 1
                bcx, bcy, bw, bh = compute_bounding_box(torch.from_numpy(gt[j, 1:].reshape(1, -1)), n=self.num_keypoints)[0]
                coco_gt.dataset["images"].append({"id": i})
                
                coords = gt[j, 1:]
                # x,y to x,y,v
                coco_keypoints = []
                for i in range(0, len(coords), 2):
                    x = coords[i]
                    y = coords[i+1]
                    coco_keypoints.extend([x,y,1])
                coco_gt.dataset["annotations"].append({"image_id": i, "category_id": gt[j, 0],
                                                    "num_keypoints": int(self.num_keypoints), 
                                                    "keypoints": coco_keypoints,
                                                    "bbox": np.hstack([bcx - 0.5 * bw, bcy - 0.5 * bh, bw, bh]),
                                                    "area": bw*bh,
                                                    "id": k, "iscrowd": 0})
                
        coco_gt.dataset["categories"] = [{"id": i, "supercategory": c, "name": c, "keypoints":["tl","bl","br","tr"]} for i, c in enumerate(self.classes)]
        coco_gt.createIndex()

        # Create preadict 
        coco_pred = COCO()
        coco_pred.dataset = {}
        coco_pred.dataset["images"] = []
        coco_pred.dataset["annotations"] = []
        k = 0
        for i, pred in enumerate(preds):
            for j in range(pred.shape[0]):
                k += 1
                bcx, bcy, bw, bh = compute_bounding_box(torch.from_numpy(pred[j, 2:].reshape(1, -1)), n=self.num_keypoints)[0]
                coords = pred[j, 2:]
                # x,y to x,y,v
                coco_keypoints = []
                for i in range(0, len(coords), 2):
                    x = coords[i]
                    y = coords[i+1]
                    coco_keypoints.extend([x,y,1])
                coco_pred.dataset["images"].append({"id": i})
                coco_pred.dataset["annotations"].append({"image_id": i, "category_id": np.int32(pred[j, 0]),
                                                        "score": pred[j, 1], 
                                                        "num_keypoints": int(self.num_keypoints),
                                                        "keypoints": coco_keypoints,
                                                        "bbox": np.hstack([bcx - 0.5 * bw, bcy - 0.5 * bh, bw, bh]),
                                                        "area": bw*bh,
                                                        "id": k})
                
        coco_pred.dataset["categories"] = [{"id": i, "supercategory": c, "name": c, "keypoints":["tl","bl","br","tr"]} for i, c in enumerate(self.classes)]
        coco_pred.createIndex()

        coco_eval = COCOeval(coco_gt, coco_pred, "keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP05 = coco_eval.stats[1]
        return mAP05

    def compute_map(self, val_dataloader, model):
        gts, pts = [], []
        pbar = tqdm(val_dataloader)

        for i, (imgs, targets) in enumerate(pbar):
            # 数据预处理
            imgs = imgs.to(self.device).float() / 255.0
            
            with torch.no_grad():
                # 模型预测
                preds = model(imgs)
                # 特征图后处理
                output = handle_preds(preds, self.device, num_keypoints=self.num_keypoints, conf_thresh=0.0001)

            # 检测结果
            N, _, H, W = imgs.shape
            
            pred_kpts = []
            for p in output:
                pbboxes = []
                for b in p:
                    b = b.cpu().numpy()
                    score = b[0]
                    category = b[1]
                    kpts = b[2:]
                    pbox = [category, score] + kpts.tolist() 
                    pbboxes.append(pbox)
                    # n_pbox += 1
                pts.append(np.array(pbboxes))
            
            # 标注结果
            for n in range(N):
                tbboxes = []
                for t in targets:
                    if t[0] == n:
                        t = t.cpu().numpy()
                        category = t[1]
                        kpts = t[4:]
                        tbox = [category] + kpts.tolist()
                        tbboxes.append(tbox)
                        # n_gt += 1
                gts.append(np.array(tbboxes))

       
        
        # print(n_gt, n_pbox)

        # mAP 暂时不可用
        mAP05 = self.coco_evaluate(gts, pts)
        

        return mAP05
