# ------------------------------------------------------------------------------------
# PatentCLIP - Image retrieval
# 
# ------------------------------------------------------------------------------------
# Modified from SWIN+ArcFace (https://github.com/L4Clippers/Patent-Image-Retrieval-Transformer-DML)
# ------------------------------------------------------------------------------------




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import open_clip

from tqdm import tqdm

import timm
import math
import random
from PIL import ImageFilter

import os.path as osp
import glob

import copy
import torch.utils.data as data
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from pytorch_metric_learning import losses, distances, regularizers


class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 1.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(data=torch.ones(1)*p, requires_grad=False) 
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'



class PatentNet(nn.Module):
    def __init__(self, model: nn.Module, embedding_size: int):
        super().__init__()
        
        # Modify the first conv layer to accept 1-channel grayscale images
        original_conv1 = model.visual.conv1
        
        # Create a new conv layer with 1 input channel
        new_conv1 = nn.Conv2d(
            in_channels=1,  # Change to 1 for grayscale images
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        
        # Copy the weights from the original conv1 for the first channel
        with torch.no_grad():
            new_conv1.weight[:, 0:1, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
        
        # Replace the original conv1 with the new one
        model.visual.conv1 = new_conv1
        
        self.model = model  # Store the modified model
        
        in_features = 512
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(),
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.PReLU(),
        )

    def forward(self, x):
        # Forward pass through the modified CLIP model
        x = self.model.encode_image(x)
        # Pass the output through the head
        x = self.head(x)
        return x

device = torch.device("cuda")

model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:patentclip/PatentCLIP_Vit_B',device=device)
#model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai',device=device)
model = PatentNet(model, embedding_size=512).to(device)
model = nn.DataParallel(model)

batch_size = 256
log_interval = 100


def train(model, metric, loss_func, device, train_loader, optimizer, metric_optimizer, epoch):
    print('---start training---')
    model.train()
    total_loss = 0
    for batch_idx, (data, labels) in enumerate(tqdm(train_loader)):
        data, labels = data.to(device), labels.to(device) # labels
        #print(labels)
        optimizer.zero_grad()
        metric_optimizer.zero_grad()
        embeddings = model(data)
        #print(data.shape)
        #print(embeddings)
        output = metric(embeddings,labels)
        #print(embeddings.size())
        #print(labels.min(), labels.max())
        loss = loss_func(output, labels)
        #print(loss)
        loss.backward()

        optimizer.step()
        metric_optimizer.step()
        total_loss += loss
        
#         if scheduler is not None:
#             scheduler.step()
        
        if batch_idx % 500 == 0:
            print("Epoch {} Iteration {}: Train Loss = {}".format(epoch, batch_idx, loss))
    
    total_loss = total_loss/len(train_loader)

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

def test(train_set, test_set, model, accuracy_calculator):
    print('---start test---')
    model.eval()
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    print(train_embeddings.shape)
    print(train_labels.shape)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    print(test_embeddings.shape)
    print(test_labels.shape)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels, train_embeddings, train_labels, False)
    print("Test set accuracy (Precision@1) = {}, (mAP) = {}".format(accuracies["precision_at_1"], accuracies["mean_average_precision"]))

def test_calc(model, metric, loss_func, device, test_query_loader, epoch):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(test_query_loader):
            data, labels = data.to(device), labels.to(device) # labels
            embeddings = model(data)
            output = metric(embeddings,labels)
            loss = loss_func(output, labels)
            total_loss += loss
            
            if batch_idx % 50 == 0:
                print("Epoch {} Iteration {}: Test Loss = {}".format(epoch, batch_idx, loss))
 
    
        total_loss = total_loss/len(test_query_loader)

def valuation(val_db_set, val_query_set, model, accuracy_calculator):
    model.eval()
    val_db_embeddings, val_db_labels = get_all_embeddings(val_db_set, model)
    val_query_embeddings, val_query_labels = get_all_embeddings(val_query_set, model)
    val_db_labels = val_db_labels.squeeze(1)
    val_query_labels = val_query_labels.squeeze(1)
#     print("Val accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        val_query_embeddings, val_db_embeddings, val_query_labels, val_db_labels, False
    )
    print("Val set accuracy, (Precision@1) = {}, (mAP) = {}".format(accuracies["precision_at_1"], accuracies["mean_average_precision"]))


def valuation_calc(model, metric, loss_func, device, val_query_loader, epoch):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch_idx, (data, labels) in enumerate(val_query_loader):
            data, labels = data.to(device), labels.to(device) # labels
            embeddings = model(data)
            output = metric(embeddings,labels)
            loss = loss_func(output, labels)
            total_loss += loss
            
            if batch_idx % 50 == 0:
                print("Epoch {} Iteration {}: Test Loss = {}".format(epoch, batch_idx, loss))
        
        total_loss = total_loss/len(val_query_loader)



data_transform = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=224),
        transforms.RandomCrop((224), pad_if_needed=True, fill=255),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([GaussianBlur()], p=0.5),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.22, 0.33), ratio=(0.3, 3.3), value=1, inplace=False), # ToTensorの後
        transforms.Normalize((0.5), (0.5)),
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(size=224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
}



def make_datapath_list(phase):
    """
    phase: train or val
    """
    

    if phase == "train":
        with open('/home/patent/IR/data/patlist/train_patent_trn.txt') as f:
            lines = f.readlines()
    elif phase == "test_query":
        with open('/home/patent/IR/data/patlist/test_query_patent.txt') as f:
            lines = f.readlines()
    elif phase == "test_db":
        with open('/home/patent/IR/data/patlist/test_db_patent.txt') as f:
            lines = f.readlines()
    elif phase == "val_query":
        with open('/home/patent/IR/data/patlist/val_query_patent.txt') as f:
            lines = f.readlines()
    elif phase == "val_db":
        with open('/home/patent/IR/data/patlist/val_db_patent.txt') as f:
            lines = f.readlines()
            
    path_list = [] # 
    img_path = '/home/patent/IR/data/patent_data/'
    for path in lines:
        path = path.split()[0] # 0:path only, 1: label
        path_list.append(img_path+path)
        
    return path_list

train_list = make_datapath_list(phase="train")

test_query_list = make_datapath_list(phase="test_query")
test_db_list = make_datapath_list(phase="test_db")

val_query_list = make_datapath_list(phase="val_query")
val_db_list = make_datapath_list(phase="val_db")

print(train_list[0:3])
print()
print("train_list:", len(train_list))
print()
print("test_query_list:", len(test_query_list))
print("test_db_list:", len(test_db_list))
print()
print("val_query_list:", len(val_query_list))
print("val_db_list", len(val_db_list))


def make_label_list(phase):
    """
    phase: train or val
    """    

    if phase == "train":
        with open('/home/patent/IR/data/patlist/train_patent_trn.txt') as f:
            lines = f.readlines()
    elif phase == "test_query":
        with open('/home/patent/IR/data/patlist/test_query_patent.txt') as f:
            lines = f.readlines()
    elif phase == "test_db":
        with open('/home/patent/IR/data/patlist/test_db_patent.txt') as f:
            lines = f.readlines()
    elif phase == "val_query":
        with open('/home/patent/IR/data/patlist/val_query_patent.txt') as f:
            lines = f.readlines()
    elif phase == "val_db":
        with open('/home/patent/IR/data/patlist/val_db_patent.txt') as f:
            lines = f.readlines()
        
    label_list = []
    for label in lines:
        label = label.split()[1] # 0:path only, 1: label
        label = int(label)
        label_list.append(label)
        
    return label_list

train_label_list = make_label_list(phase="train")

test_query_label_list = make_label_list(phase="test_query")
test_db_label_list = make_label_list(phase="test_db")

val_query_label_list = make_label_list(phase="val_query")
val_db_label_list = make_label_list(phase="val_db")

print(train_label_list[0:3])
print(type(train_label_list[0]))


def cv2pil(image):

    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

class PatentDataset(data.Dataset):
    def __init__(self, file_list, label_list, transform):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # open
        _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 
        
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnt = contours[0]
        
        x,y,w,h = cv2.boundingRect(cnt) 
        img = img[y:y+h, x:x+w] 
        
        img = cv2pil(img)
        img = self.transform(img)
        
        label = self.label_list[index]
        label = label-1 # 0-indexed [0, nclass-1]
        
        return img, label

import os
num_workers = os.cpu_count() - 1
print(num_workers)


train_data = PatentDataset(file_list=train_list, label_list=train_label_list, transform=data_transform['train'])

test_query_data = PatentDataset(file_list=test_query_list, label_list=test_query_label_list, transform=data_transform['val'])
test_db_data = PatentDataset(file_list=test_db_list, label_list=test_db_label_list, transform=data_transform['val'])

val_query_data = PatentDataset(file_list=val_query_list, label_list=val_query_label_list, transform=data_transform['val'])
val_db_data = PatentDataset(file_list=val_db_list, label_list=val_db_label_list, transform=data_transform['val'])


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True)

test_query_loader = torch.utils.data.DataLoader(test_query_data, batch_size=batch_size, num_workers=num_workers,drop_last=True)
test_db_loader = torch.utils.data.DataLoader(test_db_data, batch_size=batch_size, num_workers=num_workers,drop_last=True)

val_query_loader = torch.utils.data.DataLoader(val_query_data, batch_size=batch_size, num_workers=num_workers,drop_last=True)
val_db_loader = torch.utils.data.DataLoader(val_db_data, batch_size=batch_size, num_workers=num_workers,drop_last=True)


idx = 0
print(train_data.__getitem__(idx)[0].size())
print(train_data.__getitem__(idx)[1])
print(len(train_data))

dataloaders_dict = {"train": train_loader, "test_query": test_query_loader, "val_query": val_query_loader}

batch_iterator = iter(dataloaders_dict["train"])
inputs, labels = next(batch_iterator)

print(inputs.size())


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
    
#loss_func = nn.CrossEntropyLoss()     

class FocalLoss(nn.Module):

    def __init__(self, gamma=1.5, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

loss_func = FocalLoss()
#loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-5)
accuracy_calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision"), k="max_bin_count")
app_count = 33364

num_epochs = 25

metric = ArcMarginProduct(512, app_count, s=30.0, m=0.5, easy_margin=True).to(device)
metric_optimizer = optim.Adam(metric.parameters(), lr=5e-3)
#model.load_state_dict(torch.load('IR/clipvit-arcface-test_epoch15.pth'))


for epoch in range(1, num_epochs + 1):

    train(model, metric, loss_func, device, train_loader, optimizer, metric_optimizer, epoch)
    
    if epoch%5==0:
        torch.save(model.state_dict(), 'patentclip-arcface-test_epoch{}.pth'.format(epoch))
    
    test(test_db_data, test_query_data, model, accuracy_calculator)
    test_calc(model, metric, loss_func, device, test_query_loader, epoch)
    
    valuation(val_db_data, val_query_data, model, accuracy_calculator)
    valuation_calc(model, metric, loss_func, device, val_query_loader, epoch)
ß