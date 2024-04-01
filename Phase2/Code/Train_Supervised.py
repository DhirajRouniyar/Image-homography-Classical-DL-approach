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
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as T
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import sys
import os
import numpy as np
import argparse
import random
from tqdm import tqdm
from Model_supervised import HNet
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Img_pth(path):
    
    image_names = []
    for file in os.listdir(path):
        image_names.append(os.path.splitext(file)[0])
    
    return image_names

#Not used Cnf matrix
def Confusion_matrix(LabelsTrue,LabelsPred,x):
    #LabelsTrue = np.round(LabelsTrue).astype(int)
    #LabelsPred = np.round(LabelsPred).astype(int)
    
    #LabelsTrue_flat = LabelsTrue.flatten()
    #LabelsPred_flat = LabelsPred.flatten()
    
    #unique_classes = np.unique(np.concatenate([LabelsTrue_flat, LabelsPred_flat]))
    
    cm = metrics.confusion_matrix(y_true=LabelsTrue,  # True class for test-set.
                          y_pred=LabelsPred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(10):
        print(str(cm[i, :]) + ' ({0})'.format(i))

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(10)]
    print("".join(class_numbers))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"{i}" for i in unique_classes])
    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [f"{i}" for i in range(10)])
    cm_display.plot()
    plt.savefig(f"./Plots/train_cm_{x}.png")
    plt.show()

def Batch_gen(path_A, path_B, coordinates_path, batch_size=64):
    
    img_batch = []
    labels_batch = []
    image_num = 0
    coordinates = np.load(coordinates_path)
    
    while image_num < batch_size:
        
        image_names = Img_pth(path_A)            
        RandIdx = random.randint(0, len(image_names)-1)
        image_pathA = path_A + '/' + image_names[RandIdx] + '.jpg'
        image_pathB = path_B + '/' + image_names[RandIdx] + '.jpg'
        
        image_num += 1

    
        imgA = cv2.imread(image_pathA, cv2.IMREAD_GRAYSCALE)
        imgB = cv2.imread(image_pathB, cv2.IMREAD_GRAYSCALE)
        
        label = coordinates[RandIdx]
        
        imgA = torch.from_numpy((imgA.astype(np.float32) - 127.5) / 127.5)
        imgB = torch.from_numpy((imgB.astype(np.float32) - 127.5) / 127.5)
        
        label = torch.from_numpy(label.astype(np.float32)/32.0)
        
        img = torch.stack((imgA, imgB), dim=0)
        
        img_batch.append(img.to(device))
        labels_batch.append(label.to(device))
           
    return torch.stack(img_batch), torch.stack(labels_batch)

def prettyPrint(NumEpochs, MiniBatchSize):
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Mini Batch Size " + str(MiniBatchSize))


def train(path_ATrain, path_BTrain,
        coordinates_path,
        batch_size, 
        num_epochs, 
        CheckPointPath):
    
    torch.cuda.empty_cache()
    history = []
    model = HNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    
    num_samples_train = len(Img_pth(path_ATrain))
    
    num_iter_per_epoch = num_samples_train//batch_size
    loss_iters_training = []
    loss_iters_training_with_detach = []   #Here out of Train, Val and Test, Train and Val is same,
                                         #Diff of above two is 1st one is without loss.detach() while 2nd is with loss.detach()
    
    loss_epochs = []
    Epochs = []
    LabelTrue = []
    LabelPred = [] 
    
    for epoch in tqdm(range(num_epochs)):
        
        for iter_counter in tqdm(range(num_iter_per_epoch)):
            
            train_batch = Batch_gen(path_ATrain, path_BTrain, coordinates_path, batch_size)
            
            # Train
            model.train()
            optimizer.zero_grad()
            #Predict output with forward pass and calculate loss pred-GT
            batch_loss_train = model.training_step(train_batch)
            loss_iters_training.append(batch_loss_train) #JUST appending list for each iteration only loss no loss.detach()
            batch_loss_train.backward()
            optimizer.step()
            
            loss_iters_training_with_detach.append(model.validation_step(train_batch)) #JUST appending list for each iteration with loss.detach() 
            
            #loss_iters_training.append(batch_loss_train.detach().numpy())
            result = model.validation_step(train_batch) #Returns loss.detach() for each iteration
            model.epoch_end(epoch * num_iter_per_epoch + iter_counter, result)
            
            pred,gt = model.conf_matrixx(train_batch)

            LabelTrue.extend(gt.tolist())
            LabelPred.extend(pred.tolist())
            
            
        # Save model every epoch
        SaveName = CheckPointPath + str(epoch) + "_model.ckpt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": batch_loss_train,
            },
            SaveName,
        )
        print("\n" + SaveName + " Model Saved...")
        
        result = model.validation_end(loss_iters_training_with_detach) #Returns mean value of loss.detach() in all iterations and is a epoch loss
        #result['train_loss'] = torch.stack(loss_iters_training).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        #loss_epochs.append(np.mean(loss_iters_training))
        loss_epochs.append(torch.stack(loss_iters_training).mean().item())
        Epochs.append(epoch)
        
    
    plt.subplots(1, 2, figsize=(15,15))
        
    plt.subplot(1, 2, 1) #loss
    plt.plot(Epochs, loss_epochs)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.xlim((0,num_epochs))
    plt.show()
    return history

def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="../Data",
        help="Base path of images, Default:../Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="./Checkpoints/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=8,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:64",
    )

    Args = Parser.parse_args()
    num_epochs = Args.NumEpochs
    BasePath = Args.BasePath
    batch_size = Args.MiniBatchSize
    CheckPointPath = Args.CheckPointPath

    path_ATrain = BasePath + "/Trained_/Patch_A"
    path_BTrain = BasePath + "/Trained_/Patch_B"
    coordinates_path = BasePath + '/Trained_/h4_list.npy' 

    prettyPrint(num_epochs, batch_size)

    history = []

    history += train(path_ATrain, path_BTrain,
            coordinates_path,
            batch_size, 
            num_epochs,
            CheckPointPath)

    np.save('history.npy', np.array(history))
    #plotLosses(history)
    

    



if __name__ == "__main__":
    main()