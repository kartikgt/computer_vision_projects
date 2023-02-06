#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from skimage.util import montage
import os
import cv2
import random
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import glob
import shutil
import numpy as np
from torchvision.models import vgg19_bn, resnet50, resnet18
import numpy as np
import seaborn as sns
import albumentations as A
from torchsummary import summary

random.seed(0)

log_dir = "~/logs"
writer = SummaryWriter(log_dir)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
covid_files_path = 'Images-processed/CT_COVID'
covid_files = [os.path.join(covid_files_path, x) for x in os.listdir(covid_files_path)]
covid_images = [cv2.imread(x) for x in random.sample(covid_files, 5)]

plt.figure(figsize=(20, 10))
columns = 5
# for i, image in enumerate(covid_images):
#     plt.imshow(image)
#     plt.show()


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


class CovidCTDataset(Dataset):
    def __init__(self, root_dir, classes, covid_files, non_covid_files, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.files_path = [non_covid_files, covid_files]
        self.image_list = []

        # read the files from data split text files
        covid_files = read_txt(covid_files)
        non_covid_files = read_txt(non_covid_files)

        # combine the positive and negative files into a cummulative files list
        for cls_index in range(len(self.classes)):
            class_files = [[os.path.join(self.root_dir, self.classes[cls_index], x), cls_index] \
                           for x in read_txt(self.files_path[cls_index])]
            self.image_list += class_files

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx][0]

        # Read the image
        image = Image.open(path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = int(self.image_list[idx][1])

        data = {'img': image,
                'label': label,
                'paths': path}

        return data


normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
train_transformer_jitter = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    normalize
])
train_transformer_noise = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.GaussianBlur(5),
    transforms.ToTensor(),
    normalize
])
train_transformer_affine = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomAffine(90),
    transforms.ToTensor(),
    normalize
])
train_transformer_random_rotation = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    normalize
])
train_transformer_random_perspective = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

batchsize = 8

trainset1 = CovidCTDataset(root_dir='Images-processed/',
                          classes=['CT_NonCOVID', 'CT_COVID'],
                          covid_files='Data-split/COVID/trainCT_COVID.txt',
                          non_covid_files='Data-split/NonCOVID/trainCT_NonCOVID.txt',
                          transform=train_transformer)
trainset2 = CovidCTDataset(root_dir='Images-processed/',
                          classes=['CT_NonCOVID', 'CT_COVID'],
                          covid_files='Data-split/COVID/trainCT_COVID.txt',
                          non_covid_files='Data-split/NonCOVID/trainCT_NonCOVID.txt',
                          transform=train_transformer_noise)
trainset3 = CovidCTDataset(root_dir='Images-processed/',
                          classes=['CT_NonCOVID', 'CT_COVID'],
                          covid_files='Data-split/COVID/trainCT_COVID.txt',
                          non_covid_files='Data-split/NonCOVID/trainCT_NonCOVID.txt',
                          transform=train_transformer_jitter)
trainset4 = CovidCTDataset(root_dir='Images-processed/',
                          classes=['CT_NonCOVID', 'CT_COVID'],
                          covid_files='Data-split/COVID/trainCT_COVID.txt',
                          non_covid_files='Data-split/NonCOVID/trainCT_NonCOVID.txt',
                          transform=train_transformer_affine)
trainset5 = CovidCTDataset(root_dir='Images-processed/',
                          classes=['CT_NonCOVID', 'CT_COVID'],
                          covid_files='Data-split/COVID/trainCT_COVID.txt',
                          non_covid_files='Data-split/NonCOVID/trainCT_NonCOVID.txt',
                          transform=train_transformer_random_rotation)
trainset6 = CovidCTDataset(root_dir='Images-processed/',
                          classes=['CT_NonCOVID', 'CT_COVID'],
                          covid_files='Data-split/COVID/trainCT_COVID.txt',
                          non_covid_files='Data-split/NonCOVID/trainCT_NonCOVID.txt',
                          transform=train_transformer_random_perspective)
combined_train_set = torch.utils.data.ConcatDataset([trainset1, trainset2, trainset3, trainset4, trainset5, trainset6])
valset = CovidCTDataset(root_dir='Images-processed/',
                        classes=['CT_NonCOVID', 'CT_COVID'],
                        covid_files='Data-split/COVID/valCT_COVID.txt',
                        non_covid_files='Data-split/NonCOVID/valCT_NonCOVID.txt',
                        transform=val_transformer)
testset = CovidCTDataset(root_dir='Images-processed/',
                         classes=['CT_NonCOVID', 'CT_COVID'],
                         covid_files='Data-split/COVID/testCT_COVID.txt',
                         non_covid_files='Data-split/NonCOVID/testCT_NonCOVID.txt',
                         transform=val_transformer)
testset1 = CovidCTDataset(root_dir='Images-processed/',
                         classes=['CT_NonCOVID', 'CT_COVID'],
                         covid_files='Data-split/COVID/testCT_COVID.txt',
                         non_covid_files='Data-split/NonCOVID/testCT_NonCOVID.txt',
                         transform=train_transformer_noise)
testset2 = CovidCTDataset(root_dir='Images-processed/',
                         classes=['CT_NonCOVID', 'CT_COVID'],
                         covid_files='Data-split/COVID/testCT_COVID.txt',
                         non_covid_files='Data-split/NonCOVID/testCT_NonCOVID.txt',
                         transform=train_transformer_affine)
testset3 = CovidCTDataset(root_dir='Images-processed/',
                         classes=['CT_NonCOVID', 'CT_COVID'],
                         covid_files='Data-split/COVID/testCT_COVID.txt',
                         non_covid_files='Data-split/NonCOVID/testCT_NonCOVID.txt',
                         transform=train_transformer_random_rotation)
testset4 = CovidCTDataset(root_dir='Images-processed/',
                         classes=['CT_NonCOVID', 'CT_COVID'],
                         covid_files='Data-split/COVID/testCT_COVID.txt',
                         non_covid_files='Data-split/NonCOVID/testCT_NonCOVID.txt',
                         transform=train_transformer_jitter)
testset5 = CovidCTDataset(root_dir='Images-processed/',
                         classes=['CT_NonCOVID', 'CT_COVID'],
                         covid_files='Data-split/COVID/testCT_COVID.txt',
                         non_covid_files='Data-split/NonCOVID/testCT_NonCOVID.txt',
                         transform=train_transformer_random_perspective)
combined_test_set = torch.utils.data.ConcatDataset([testset1, testset2, testset3, testset4, testset5, testset])
train_loader = DataLoader(trainset1, batch_size=batchsize, drop_last=False, shuffle=True)
val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
test_loader = DataLoader(combined_test_set, batch_size=batchsize, drop_last=False, shuffle=False)


def compute_metrics(model, test_loader, plot_roc_curve=False):
    model.eval()

    val_loss = 0
    val_correct = 0

    criterion = nn.CrossEntropyLoss()

    score_list = torch.Tensor([]).to(device)
    pred_list = torch.Tensor([]).to(device).long()
    target_list = torch.Tensor([]).to(device).long()
    path_list = []

    for iter_num, data in enumerate(test_loader):
        # Convert image data into single channel data
        image, target = data['img'].to(device), data['label'].to(device)
        paths = data['paths']
        path_list.extend(paths)

        # Compute the loss
        with torch.no_grad():
            output = model(image)

        # Log loss
        val_loss += criterion(output, target.long()).item()

        # Calculate the number of correctly classified examples
        pred = output.argmax(dim=1, keepdim=True)
        val_correct += pred.eq(target.long().view_as(pred)).sum().item()

        # Bookkeeping
        score_list = torch.cat([score_list, nn.Softmax(dim=1)(output)[:, 1].squeeze()])
        pred_list = torch.cat([pred_list, pred.squeeze()])
        target_list = torch.cat([target_list, target.squeeze()])

    classification_metrics = classification_report(target_list.tolist(), pred_list.tolist(),
                                                   target_names=['CT_NonCOVID', 'CT_COVID'],
                                                   output_dict=True)

    # sensitivity is the recall of the positive class
    sensitivity = classification_metrics['CT_COVID']['recall']

    # specificity is the recall of the negative class
    specificity = classification_metrics['CT_NonCOVID']['recall']

    # accuracy
    accuracy = classification_metrics['accuracy']

    # confusion matrix
    conf_matrix = confusion_matrix(target_list.tolist(), pred_list.tolist())

    # roc score
    roc_score = roc_auc_score(target_list.tolist(), score_list.tolist())

    # plot the roc curve
    if plot_roc_curve:
        fpr, tpr, _ = roc_curve(target_list.tolist(), score_list.tolist())
        plt.plot(fpr, tpr, label="Area under ROC = {:.4f}".format(roc_score))
        plt.legend(loc='best')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    # put together values
    metrics_dict = {"Accuracy": accuracy,
                    "Sensitivity": sensitivity,
                    "Specificity": specificity,
                    "Roc_score": roc_score,
                    "Confusion Matrix": conf_matrix,
                    "Validation Loss": val_loss / len(test_loader),
                    "score_list": score_list.tolist(),
                    "pred_list": pred_list.tolist(),
                    "target_list": target_list.tolist(),
                    "paths": path_list}

    return metrics_dict


model = resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)
model.to(device)
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


class EarlyStopping(object):
    def __init__(self, patience=8):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.previous_loss = int(1e8)
        self.previous_accuracy = 0
        self.init = False
        self.accuracy_decrease_iters = 0
        self.loss_increase_iters = 0
        self.best_running_accuracy = 0
        self.best_running_loss = int(1e7)

    def add_data(self, model, loss, accuracy):

        # compute moving average
        if not self.init:
            running_loss = loss
            running_accuracy = accuracy
            self.init = True

        else:
            running_loss = 0.2 * loss + 0.8 * self.previous_loss
            running_accuracy = 0.2 * accuracy + 0.8 * self.previous_accuracy

        # check if running accuracy has improved beyond the best running accuracy recorded so far
        if running_accuracy < self.best_running_accuracy:
            self.accuracy_decrease_iters += 1
        else:
            self.best_running_accuracy = running_accuracy
            self.accuracy_decrease_iters = 0

        # check if the running loss has decreased from the best running loss recorded so far
        if running_loss > self.best_running_loss:
            self.loss_increase_iters += 1
        else:
            self.best_running_loss = running_loss
            self.loss_increase_iters = 0

        # log the current accuracy and loss
        self.previous_accuracy = running_accuracy
        self.previous_loss = running_loss

    def stop(self):

        # compute thresholds
        accuracy_threshold = self.accuracy_decrease_iters > self.patience
        loss_threshold = self.loss_increase_iters > self.patience

        # return codes corresponding to exhuaustion of patience for either accuracy or loss
        # or both of them
        if accuracy_threshold and loss_threshold:
            return 1

        if accuracy_threshold:
            return 2

        if loss_threshold:
            return 3

        return 0

    def reset(self):
        # reset
        self.accuracy_decrease_iters = 0
        self.loss_increase_iters = 0


early_stopper = EarlyStopping(patience=5)

best_model = model
best_val_score = 0

criterion = nn.CrossEntropyLoss()

for epoch in range(60):

    model.train()
    train_loss = 0
    train_correct = 0

    for iter_num, data in enumerate(train_loader):
        image, target = data['img'].to(device), data['label'].to(device)

        # Compute the loss
        output = model(image)
        loss = criterion(output, target.long()) / 8

        # Log loss
        train_loss += loss.item()
        loss.backward()

        # Perform gradient udpate
        if iter_num % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Calculate the number of correctly classified examples
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.long().view_as(pred)).sum().item()

    # Compute and print the performance metrics
    metrics_dict = compute_metrics(model, val_loader)
    print('------------------ Epoch {} Iteration {}--------------------------------------'.format(epoch,
                                                                                                  iter_num))
    print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
    print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
    print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
    print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
    print("Val Loss \t {}".format(metrics_dict["Validation Loss"]))
    print("------------------------------------------------------------------------------")

    with open('out_res_noise.txt', 'w') as f:
        print('------------------ Epoch {} Iteration {}--------------------------------------'.format(epoch,
                                                                                                      iter_num))
        print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
        print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
        print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
        print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
        print("Val Loss \t {}".format(metrics_dict["Validation Loss"]))
        print("------------------------------------------------------------------------------")

    # Save the model with best validation accuracy
    # if metrics_dict['Accuracy'] > best_val_score:
    #     torch.save(model, "best_model.pkl")
    #     best_val_score = metrics_dict['Accuracy']

    # print the metrics for training data for the epoch
    print('\nTraining Performance Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
               100.0 * train_correct / len(train_loader.dataset)))

    with open('out_res_noise.txt', 'w') as f:
        print('\nTraining Performance Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
                   100.0 * train_correct / len(train_loader.dataset)))

    # log the accuracy and losses in tensorboard
    writer.add_scalars("Losses", {'Train loss': train_loss / len(train_loader),
                                  'Validation_loss': metrics_dict["Validation Loss"]},
                       epoch)
    writer.add_scalars("Accuracies", {"Train Accuracy": 100.0 * train_correct / len(train_loader.dataset),
                                      "Valid Accuracy": 100.0 * metrics_dict["Accuracy"]}, epoch)

    # Add data to the EarlyStopper object
    early_stopper.add_data(model, metrics_dict['Validation Loss'], metrics_dict['Accuracy'])

    # If both accuracy and loss are not improving, stop the training
    if early_stopper.stop() == 1:
        break

    # if only loss is not improving, lower the learning rate
    if early_stopper.stop() == 3:
        for param_group in optimizer.param_groups:
            learning_rate *= 0.1
            param_group['lr'] = learning_rate
            print('Updating the learning rate to {}'.format(learning_rate))
            early_stopper.reset()

#model = torch.load("best_model.pkl")

metrics_dict = compute_metrics(model, test_loader, plot_roc_curve=True)
print('------------------- Test Performance --------------------------------------')
print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
print("------------------------------------------------------------------------------")

with open('out_res_noise.txt', 'w') as f:
    print('------------------- Test Performance --------------------------------------')
    print("Accuracy \t {:.3f}".format(metrics_dict['Accuracy']))
    print("Sensitivity \t {:.3f}".format(metrics_dict['Sensitivity']))
    print("Specificity \t {:.3f}".format(metrics_dict['Specificity']))
    print("Area Under ROC \t {:.3f}".format(metrics_dict['Roc_score']))
    print("------------------------------------------------------------------------------")