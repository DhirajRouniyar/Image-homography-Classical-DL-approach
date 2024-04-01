"""
Spring 2024: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: Autopano

Author(s):
Dhiraj Kumar Rouniyar (dkrouniyar@wpi.edu)
MS Robotics,
Worcester Poytechnic Insitute, MA

Dhrumil Sandeep Kotadia (dkotadia@wpi.edu)
MS Robotics,
Worcester Poytechnic Insitute, MA
"""

import torch
import cv2
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import kornia
import numpy as np
from kornia.geometry.transform import HomographyWarper as HomographyWarper
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision.transforms as T


def lossFn(warped_a, b):
    loss = (b - warped_a).float().mean()
    loss = torch.tensor(loss, requires_grad=True)
    return loss

def getLoss(pred, labels):
    criterion = nn.MSELoss()
    loss = criterion(pred, labels.view(-1, 8))
    return loss

def warpImage(I_A, H):
    warper = HomographyWarper(I_A.shape[2], I_A.shape[3])
    w = warper(torch.ones(1, 1, 420, 526), torch.eye(3).unsqueeze(0))
    return w
    
def Batch_gen(batch, H_batch):
    p_ab, _, corners_a, imgA_paths = batch
    PA_warped_lst = []
    
    for path, h, corner in zip(imgA_paths, H_batch, corners_a):
        
        Y_C = int(corner[0][1])
        Y1_C = int(corner[1][1])

        X_C = int(corner[0][0])
        X1_C = int(corner[2][0])

        I_A = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        I_A = torch.from_numpy(I_A).to(torch.float)
        I_A = I_A.unsqueeze(0).unsqueeze(0)
    
        h=torch.FloatTensor(h).unsqueeze(0).to(device)

        IA_warped = warpImage(I_A, h)
       
        PA_warped = IA_warped[:, :, Y_C:Y1_C, X_C:X1_C]
        
        PA_warped = PA_warped.to(device)
        PA_warped = (PA_warped-127.5)/127.5
        
        PA_warped = PA_warped.squeeze(0).squeeze(0)
        PA_warped_lst.append(PA_warped)
    
    p_b = p_ab[:, 1, :, :]
    
    return torch.stack(PA_warped_lst), p_b

def tensor_DLT(CO_A, h4pt):
    H_batch = []
    batch_size = CO_A.shape[0]
    CO_B = CO_A + h4pt.view(batch_size, 4, 2) * 32.
    
    H = torch.tensor((batch_size, 3, 3)).to(device)
    for img in range(batch_size):
        COO_A = CO_A[img, :, :]
        COO_B = CO_B[img, :, :]
        A = []
        b = []
        for i in range(4):
            a = [ [0, 0, 0, -COO_A[i, 0], -COO_A[i, 1], -1, COO_B[i, 1]*COO_A[i, 0], COO_B[i, 1]*COO_A[i, 1]], 
                    [COO_A[i, 0], COO_A[i, 1], 1, 0, 0, 0, -COO_B[i, 0]*COO_A[i, 0], -COO_B[i, 0]*COO_A[i, 1]] ]
            rhs = [[-COO_B[i, 1]], [COO_B[i, 0]]]

            A.append(a)
            b.append(rhs)
        A = torch.tensor(A, dtype=torch.float32, requires_grad=False).to(device).reshape(8, 8)
        b = torch.tensor(b, dtype=torch.float32, requires_grad=False).to(device).reshape(8, 1)
        x = torch.linalg.solve(A, b)
        h = torch.cat((x, torch.ones(1,1).to(device)), 0).to(device)
        H = h.view(3,3)
        H_batch.append(H)
    
    return torch.stack(H_batch)



class ModelBase(nn.Module):

    def training_step(self, batch):
        # print('model.training_step')
        p_ab, delta, c_a, imgA_paths = batch
        
        #print(c_a.shape, p_ab.shape)
        h4pt = self(p_ab)
        H_batch = tensor_DLT(c_a, h4pt)
        p_a_warped, p_b = Batch_gen(batch, H_batch)
        # print(p_a_warped.shape, p_b.shape)
        loss = lossFn(p_a_warped, p_b)
        return loss
    
    def validation_step(self, batch):
        p_ab, delta, c_a, imgA_paths = batch
        h4pt = self(p_ab)
        H_batch = tensor_DLT(c_a, h4pt)
        p_a_warped, p_b = Batch_gen(batch, H_batch)
        loss = lossFn(p_a_warped, p_b)
        return {'train_loss': loss.detach()}
    
    def validation_end(self, outputs):
        batch_losses = [x['train_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'train_loss': epoch_loss.item()}
    '''
    def conf_matrixx(self,batch):
        images,labels = batch
        out = self(images)                  # Generate predictions
        _, preds = torch.max(out, dim=1)
        return preds,labels
    '''
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}".format(
            epoch, result['train_loss']))

class HNet(ModelBase):
    def __init__(self):
        super(HNet,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(2,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
                                    
        self.layer2 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())        
        self.layer6 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.layer7 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(128,128,3,padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.fc1 = nn.Linear(128*16*16,1024)
        self.fc2 = nn.Linear(1024,8)
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.view(-1,128* 16* 16)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = HNet().to(device)
summary(model, (2, 128, 128))
