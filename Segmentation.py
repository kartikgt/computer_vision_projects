# This is a sample Python script.
import os.path

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import VOCSegmentation
import numpy as np
import cv2
import torch.utils.data as data
from PascalDataset import ImageData, VOC_COLORS
from tqdm import tqdm
from torchsummary import summary
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
from torchvision import transforms
from DataPls import SegmentationDataset
from torchviz import make_dot


def decode_image(label):  # 1x512x512

    label = label.to('cpu').detach()

    # make HxWxC
    label = torch.squeeze(label, 0)
    label = torch.argmax(label, dim=0).numpy()

    h, w = label.shape

    # initalize output image
    res = np.zeros((h, w, 3), dtype=int)

    # create class pixel values
    classes = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
               [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128], [0, 0, 64],
               [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192],
               [128, 128, 64], [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0],
               [0, 192, 128], [128, 64, 0]]

    for i in range(len(label)):
        for j in range(len(label[i])):
            res[i][j] = VOC_COLORS[label[i][j]]

    return res


torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
# load pretrained models, using ResNeSt-50 as an example
model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
model.eval()
summary(model, (3, 224, 224))


class Decode(torch.nn.Module):
    def __init__(self):
        super(Decode, self).__init__()
        self.deconv1 = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=2048,
                            out_channels=1024,
                            kernel_size=3,
                            stride=1,
                            padding=1
                            ),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU()
        )

        self.deconv2 = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=1024,
                            out_channels=512,
                            kernel_size=3,
                            stride=1,
                            padding=1
                            ),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )

        self.deconv3 = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=512,
                            out_channels=256,
                            kernel_size=3,
                            stride=1,
                            padding=1
                            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.deconv4 = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=256,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1
                            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU()
        )

        self.deconv5 = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=128,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1
                            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.a1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,
                            out_channels=22,
                            kernel_size=3,
                            stride=1,
                            padding=1
                            ),
            torch.nn.BatchNorm2d(22),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.a1(x)
        return x


# model.avgpool = DeepLabHead(2048, 22)
# model.avgpool = torch.nn.Identity()
model.avgpool = Decode()
# model.fc = DeepLabHead(64, 22)
# class a1(torch.nn.Module):
#     def __init__(self):
#         super(a1, self).__init__()
#         self.a1 = torch.nn.Sequential(
#             torch.nn.Conv2d(in_channels=64,
#                     out_channels=22,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1
#                     ),
#             # torch.nn.BatchNorm2d(22),
#             # torch.nn.ReLU()
#         )
#
#     def forward(self, x):
#         return self.a1(x)
# model.fc = a1()
for param in model.parameters():
    param.requires_grad = False
for param in model.avgpool.parameters():
    param.requires_grad = True
model = torch.nn.Sequential(*(list(model.children())[:-1]))
dataset = ImageData(isTrain=True)
train_loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
weights = torch.tensor([1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                       dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
EPOCHS = 20

net_state_dict = "./TrainedNet6_50"
# net = SegmentNet2().type(torch.cuda.FloatTensor)
model.load_state_dict(torch.load(net_state_dict))
model.eval()

for epoch in range(1, EPOCHS + 1):
    # model.train()
    for i, (x, y) in enumerate(tqdm(train_loader)):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        # inputImg, labelMasks = x, y
        inputs = x.to(device)
        masks = y.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, masks.type(torch.cuda.LongTensor))
        loss.backward()
        optimizer.step()
        if i % 5 == 4:
            print("Epoch: %d    Training Loss: %.5f\n" % (epoch, loss.item()))
            # wandb.log({"Training Loss": running_loss})
            torch.cuda.empty_cache()
    if epoch % 10 == 0:
        for g in optimizer.param_groups:
            g['lr'] *= 0.5
    if epoch % 5 == 0:
        torch.save(model.state_dict(), "./TrainedNet6_" + str(epoch))

