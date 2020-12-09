# HW3
# Instance segmentation

Code for homework 3 in the Selected Topics in Visual Recognition using Deep Learning.
I choose the Mask R-CNN as my network and apply the transfer learning to speed up the training process.
## Catalog
- [Instance segmentation](#instance-segmentation)
  - [Catalog](#Catalog)
  - [Installation](#Installation)
    - [Dataset](#Dataset)
    - [Requirements](#Requirements)
    - [Pretrained model](#Pretrained-model)
  - [Train](#train)
    - [Train model](#train-model)
  - [Test](#test)

## Installation
### Dataset
The dataset for this homework is here:
  - [Link](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK?usp=sharing)
### Requirements
- Python >= 3.6
- PyTorch >= 1.3.0
### Pretrained model
The final model I traiined is here:
  -  [Final Model](https://pan.baidu.com/s/184g9QWYgCMAeid_zHmdB4g) extraction code: orwy


## Train
### Train model
To train models, run following commands.
```
$ python train.py --use-cuda --iters -1 --dataset coco --epochs 40
```
The pretrained model on the ImageNet is loaded. 

Note that all hyper parameters are set done in the train.py.

## Test
Run following commands to generate the submission.json file of the testing results.
```
$ python test.py
```
