import bisect
import glob
import os
import re
import time
import torch
import pytorch_mask_rcnn as pmr
from PIL import Image
from torchvision.transforms import transforms
import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np


    
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if device.type == "cuda": 
        pmr.get_gpu_prop(show=True)
    print("\ndevice: {}".format(device))
        
    # ---------------------- prepare data loader ------------------------------- #
    
    dataset_train = pmr.datasets(args.dataset, args.data_dir, "train_images", train=True)
    indices = torch.randperm(len(dataset_train)).tolist()
    d_train = torch.utils.data.Subset(dataset_train, indices)
            
    args.warmup_iters = max(1000, len(d_train))
    
    # -------------------------------------------------------------------------- #

    print(args)
    num_classes = len(d_train.dataset.classes) + 1 # including background class
    model = pmr.maskrcnn_resnet50(num_classes).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    start_epoch = 0
    prefix, ext = os.path.splitext(args.ckpt_path)
    ckpts = glob.glob(prefix + "-*" + ext)
    ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
    if ckpts:
        checkpoint = torch.load(ckpts[-1], map_location=device) # load last checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epochs"]
        del checkpoint
        torch.cuda.empty_cache()

    since = time.time()
    print("\nalready trained: {} epochs; to {} epochs".format(start_epoch, args.epochs))
    
    # ------------------------------- train ------------------------------------ #
        
    for epoch in range(start_epoch, args.epochs):
        print("\nepoch: {}".format(epoch + 1))
            
        A = time.time()
        args.lr_epoch = args.lr
        print("lr_epoch: {:.8f}".format(args.lr_epoch))
        iter_train = pmr.train_one_epoch(model, optimizer, d_train, device, epoch, args)
        A = time.time() - A
        
        B = time.time()
        B = time.time() - B

        trained_epoch = epoch + 1
        print("training: {:.2f} s, evaluation: {:.2f} s".format(A, B))

        if (epoch+1)%10==0:
            pmr.save_ckpt(model, optimizer, trained_epoch, args.ckpt_path)


        # it will create many checkpoint files during training, so delete some.
        prefix, ext = os.path.splitext(args.ckpt_path)
        ckpts = glob.glob(prefix + "-*" + ext)
        ckpts.sort(key=lambda x: int(re.search(r"-(\d+){}".format(ext), os.path.split(x)[1]).group(1)))
        n = 5
        if len(ckpts) > n:
            for i in range(len(ckpts) - n):
                os.system("rm {}".format(ckpts[i]))

        # --------------------------------- AP ------------------------------------ #
        model.eval()
        cocoGt = COCO(args.data_dir+"pascal_train.json")
        coco_dt_train = []
        jishu = 0
        for imgid in cocoGt.imgs:
            img_id = int(imgid)
            img_info = cocoGt.imgs[img_id]
            jishu+=1
            if jishu%10 == 0:
                print(jishu)
            image = Image.open(os.path.join(args.data_dir+'train_images', img_info["file_name"]))
            image.convert("RGB")
            transform1 = transforms.Compose([transforms.ToTensor()])
            image = transform1(image)
            image = image.to(device)
            with torch.no_grad():
                result = model(image)
            # binarize the mask
            masks = result['masks'].gt_(0.5)
            masks = masks.cpu().numpy()

            categories = result['labels']
            scores = result['scores']
            n_instances = len(scores)
            if len(categories) > 0: # If any objects are detected in this image
                for i in range(n_instances): # Loop all instances
                    pred = {}
                    pred['image_id'] = imgid # this imgid must be same as the key of test.json
                    pred['category_id'] = int(categories[i])+1
                    segmentation = masks[i,:,:]
                    rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
                    rle['counts'] = rle['counts'].decode('ascii')
                    pred['segmentation'] = rle
                    pred['score'] = float(scores[i])
                    coco_dt_train.append(pred)
            else:
                pred = {'image_id':imgid,'category_id':0,'score':0.0,
                        'segmentation':{'counts':'00','size':[masks.shape[1],masks.shape[2]]}}
                coco_dt_train.append(pred)

        print(len(coco_dt_train))
        with open("/output/train"+str(epoch)+".json", "w") as f:
            json.dump(coco_dt_train, f)
        cocoDt = cocoGt.loadRes("/output/train"+str(epoch)+".json")
        cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


    # -------------------------------------------------------------------------- #

    print("\ntotal time of this training: {:.2f} s".format(time.time() - since))
    if start_epoch < args.epochs:
        print("already trained: {} epochs\n".format(trained_epoch))
        

    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cuda", action="store_true")
    
    parser.add_argument("--dataset", default="coco", help="coco or voc")
    parser.add_argument("--data-dir", default="/data/zihaosh/hw3/")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--results")
    
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument('--lr-steps', nargs="+", type=int, default=[22, 26])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--iters", type=int, default=-1, help="max iters per epoch, -1 denotes auto")
    parser.add_argument("--print-freq", type=int, default=10, help="frequency of printing losses")
    args = parser.parse_args()
    
    if args.lr is None:
        args.lr = 0.02 * 1 / 16
    if args.ckpt_path is None:
        args.ckpt_path = "/output/maskrcnn_{}.pth".format(args.dataset)
    if args.results is None:
        args.results = os.path.join(os.path.dirname(args.ckpt_path), "results.pth")
    
    main(args)
    
    