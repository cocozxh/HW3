import torch
import matplotlib.pyplot as plt
import pytorch_mask_rcnn as pmr
from PIL import Image
import os
from torchvision.transforms import transforms
import json
import numpy as np
import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

use_cuda = True
dataset = "coco"
ckpt_path = "/data/zihaosh/hw2_load/hw3-xco-60.pth"
data_dir = "/data/zihaosh/hw3/"

device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
if device.type == "cuda":
    pmr.get_gpu_prop(show=True)
print("\ndevice: {}".format(device))

ds = pmr.datasets(dataset, data_dir, "test_images", train=False)

model = pmr.maskrcnn_resnet50(len(ds.classes) + 1).to(device)
model.eval()
model.head.score_thresh = 0.3

if ckpt_path:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    torch.cuda.empty_cache()

for p in model.parameters():
    p.requires_grad_(False)

cocoGt = COCO(data_dir + "test.json")
coco_dt = []
jishu = 0
for imgid in cocoGt.imgs:
    img_id = int(imgid)
    img_info = cocoGt.imgs[img_id]
    jishu += 1
    if jishu % 10 == 0:
        print(jishu)
    image = Image.open(os.path.join(data_dir, 'test_images', img_info["file_name"]))
    image.convert("RGB")
    transform1 = transforms.Compose([transforms.ToTensor()])
    image = transform1(image)
    image = image.to(device)
    with torch.no_grad():
        result = model(image)

    #     plt.figure(figsize=(12, 15))
    #     pmr.show(image, result, ds.classes)

    masks = result['masks'].gt_(0.5)
    masks = masks.cpu().numpy()

    categories = result['labels']
    scores = result['scores']
    n_instances = len(scores)
    if len(categories) > 0:  # If any objects are detected in this image
        for i in range(n_instances):  # Loop all instances
            # save information of the instance in a dictionary then append on coco_dt list
            pred = {}
            pred['image_id'] = imgid  # this imgid must be same as the key of test.json
            pred['category_id'] = int(categories[i]) + 1
            segmentation = masks[i, :, :]
            rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('ascii')
            pred['segmentation'] = rle
            pred['score'] = float(scores[i])
            coco_dt.append(pred)

print(len(coco_dt))
with open("/output/0616109_.json", "w") as f:
    json.dump(coco_dt, f)
