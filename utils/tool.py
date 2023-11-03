import yaml
import torch
import torchvision

# 解析yaml配置文件
class LoadYaml:
    def __init__(self, path):
        with open(path, encoding='utf8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        self.val_txt = data["DATASET"]["VAL"]
        self.train_txt = data["DATASET"]["TRAIN"]
        self.names = data["DATASET"]["NAMES"]

        self.learn_rate = data["TRAIN"]["LR"]
        self.batch_size = data["TRAIN"]["BATCH_SIZE"]
        self.milestones = data["TRAIN"]["MILESTIONES"]
        self.end_epoch = data["TRAIN"]["END_EPOCH"]
        
        self.input_width = data["MODEL"]["INPUT_WIDTH"]
        self.input_height = data["MODEL"]["INPUT_HEIGHT"]
        
        self.category_num = data["MODEL"]["NC"]
        self.num_keypoints = data["MODEL"]["NPTS"]
        
        print("Load yaml sucess...")

def compute_bounding_box(keypoints: torch.Tensor, n=4) -> torch.Tensor:
    
    try:
      keypoints = keypoints.reshape(-1, n, 2)
    except(RuntimeError):
      print(keypoints)
      

    min_x = torch.min(keypoints[..., 0], dim=1).values
    min_y = torch.min(keypoints[..., 1], dim=1).values
    max_x = torch.max(keypoints[..., 0], dim=1).values
    max_y = torch.max(keypoints[..., 1], dim=1).values
    
    x = (min_x + max_x) / 2
    y = (min_y + max_y) / 2
    w = max_x - min_x
    h = max_y - min_y
    
    bboxes = torch.stack([x, y, w, h], dim=1)
    return bboxes
    
def handle_preds(preds: torch.Tensor, device, num_keypoints: int, conf_thresh=0.6, nms_thresh=0.45):
  '''
  return [[obj_score, category, keypoints],...] 
  '''
  total_objects, output_objects = [], []
  
  N, C, H, W = preds.shape
  feature_dim = int(1 + num_keypoints*2 + 1)
  
  objects = torch.zeros((N, H, W, feature_dim))
  
  pred = preds.permute(0, 2, 3, 1)
  # 前背景分类分支
  pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
  # 关键点回归分支
  preg = pred[:, :, :, 1 : 1+num_keypoints*2]
  # 目标分类分支
  pcls = pred[:, :, :, 1+num_keypoints*2:]
  
  # 置信度
  # objects[:, :, :, -1] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(dim=-1)[0] ** 0.4)
  objects[:, :, :, -1] = pobj.squeeze(-1)
  objects[:, :, :, -2] = pcls.argmax(dim=-1)
   
  # 坐标
  gy, gx = torch.meshgrid(torch.arange(H), torch.arange(W))
  for i in range(0, num_keypoints*2, 2):
    objects[:, :, :, i] = (preg[..., i] + gx.to(device)) / W
    objects[:, :, :, i+1] = (preg[..., i+1] + gy.to(device)) / H
    
  objects = objects.reshape(N, H*W, feature_dim)
  total_objects.append(objects)
  
  batch_objects = torch.cat(total_objects, dim=1)
  
  for objects in batch_objects:
    output, temp = [], []
    b, s, c = [], [], []
    kpts = []
    
    t = objects[:, -1] > conf_thresh
    tobjects = objects[t]
    for object in tobjects:
      
      obj_score = object[-1]
      category = object[-2]

      bcx, bcy, bw, bh = compute_bounding_box(object[:-2], num_keypoints)[0]
      tlx, tly = bcx - bw / 2, bcy - bh / 2
      brx, bry = bcx + bw / 2, bcy + bh / 2
      
      s.append([obj_score])
      c.append([category])
      b.append([tlx, tly, brx, bry])
      kpts.append([object[:-2]])
      
      temp.append(torch.cat((obj_score.unsqueeze(0), category.unsqueeze(0), object[:-2])))
      
    if len(b) > 0:
      b = torch.tensor(b).to(device)
      c = torch.tensor(c).squeeze(1).to(device)
      s = torch.tensor(s).squeeze(1).to(device)
      keep = torchvision.ops.nms(b, s, nms_thresh)
      
      for i in keep:
        output.append(temp[i])
        
    output_objects.append(output)
    
  return output_objects
      
      
      
      
  
    