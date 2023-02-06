import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
from PIL import Image
from torchvision import transforms

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

width = 256
height = 256
def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
    'void'
]
nclasses = 21
row_size = 50
col_size = 500
cmap = color_map()
VOC_COLORS = np.zeros([nclasses+1, 3])
VOC_COLORS[:nclasses, :] = cmap[:nclasses, :]
VOC_COLORS[-1, :] = cmap[-1,:]
# np.where(np.all(VOC_COLORS == [0,0,0], axis=1))

def convertToSegmentationMask(mask):
    height, width = mask.shape[:2]
    segmentation_mask = np.zeros((height, width, len(VOC_COLORS)), dtype=np.float32)
    segmentation_mask = np.zeros((height, width))
    for label_index, label in enumerate(VOC_COLORS):
        # segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        # segmentation_mask[:, :, label_index] = np.where(np.all(mask == label, axis=-1), 1, 0)
        segmentation_mask[:, :] = np.where(np.all(mask == label, axis=-1), label_index, segmentation_mask)
    return segmentation_mask

#Dirs created by DataSeparate.py
img_path_train = "./VOCdevkit/VOC2012/SegmentTrain"
img_path_val = "./VOCdevkit/VOC2012/SegmentVal"
label_path_train = "./VOCdevkit/VOC2012/SegmentTrainLabels"
label_path_val = "./VOCdevkit/VOC2012/SegmentValLabels"

segPath = os.path.join('.', 'VOC Dataset', 'VOCtrainval_11-May-2012', 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation')
trainTxt = os.path.join(segPath, 'train.txt')
valTxt = os.path.join(segPath, 'val.txt')
srcClassPath = os.path.join('.', 'VOC Dataset', 'VOCtrainval_11-May-2012', 'VOCdevkit', 'VOC2012', 'SegmentationClass')
srcImagePath = os.path.join('.', 'VOC Dataset', 'VOCtrainval_11-May-2012', 'VOCdevkit', 'VOC2012', 'JPEGImages')

class ImageData(Dataset):
    def __init__(self, isTrain=True):
        if isTrain:
            path = trainTxt
        else:
            path = valTxt
        my_file = open(path, "r")

        # reading the file
        data = my_file.read()

        # replacing end of line('/n') with ' ' and
        # splitting the text it further when '.' is seen.
        self.imageList = data.replace('\n', ' ').split(" ")

        # printing the data
        # print(data_into_list)
        my_file.close()

    def __len__(self):
        return (len(self.imageList))

    def __getitem__(self, idx):
        # print(os.path.join(srcClassPath, self.imageList[idx]) + '.png')
        mask = cv2.cvtColor(cv2.imread(os.path.join(srcClassPath, self.imageList[idx]) + '.png'), cv2.COLOR_BGR2RGB)
        # Convert to channels x height x width
        mask = convertToSegmentationMask(mask)
        # mask = np.swapaxes(mask, 2, 1)
        # mask = np.swapaxes(mask, 1, 0)
        mask = torch.Tensor(mask)
        mask = mask.unsqueeze(0)
        img = Image.open(os.path.join(srcImagePath, self.imageList[idx]) + '.jpg')
        img = img.convert('RGB')

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        preprocessMask = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        img = preprocess(img)
        mask2 = preprocessMask(mask)
        mask2 = mask2.squeeze(0)

        return img, mask2