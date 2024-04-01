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
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as T
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from Model_unsupervised import HNet



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
model = HNet()
checkpoint = torch.load('./Checkpoints_Unsup/9_model.ckpt')  #Load the Unsupervised trained model
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

#Draw true and predicted corners

def drw_cor(image, corners, color):
    corners = np.array(corners.copy())
    #Swap the second and third corners directly within the array
    corners[2, :], corners[3, :] = corners[3, :].copy(), corners[2, :].copy()
    
    corners = corners.reshape(-1, 1, 2).astype(int)
    
    # Draw the polylines on the image
    img_cor = cv2.polylines(image.copy(), [corners], True, color, 4)
    return img_cor

#Image preprocessing
def image_preprocessing(images):
    img_1 = cv2.resize(images[0], (128, 128), interpolation = cv2.INTER_AREA)
    img_1 =  img_1.astype(np.float32)
    img_1 = torch.from_numpy(cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY))
    
    img_2 = cv2.resize(images[1], (128, 128), interpolation = cv2.INTER_AREA)
    img_2 = img_2.astype(np.float32)
    img_2 = torch.from_numpy(cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY))

    img = torch.stack((img_1, img_2), dim=0)

    return img

#Predicting Homography
def predict_homographies(images):

    img = image_preprocessing(images)
    #print(np.shape(img))
    img = (img).float().unsqueeze(0)
    #print(np.shape(image))
    with torch.no_grad():
            output = model(img)
            
    homography = output.squeeze().numpy()
    np.save("H4pt_pred_Unsup.npy",homography)
    return homography

def main():
    
    #Get an image for Testing
    image = cv2.imread(f'../../Data/Train/9.jpg') #Read image manually to get the results
    h, w, c = image.shape

    # Select random patch size
    psi = 32
    center_x = w // 2
    center_y = h // 2

# Cut the 128x128 ROI from the center of the image
    patch1 = image[center_y-64:center_y+64, center_x-64:center_x+64]
    corner_points1=[[center_x-64,center_y-64], [center_x+64,center_y-64] , [center_x-64,center_y+64], [center_x+64,center_y+64]]

    perturbation = np.random.randint(-psi, psi + 1, (4, 2))

# Add perturbation to corner points
    perturbed_corner_points = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(corner_points1, perturbation)]
    H4pt_True = np.array(perturbed_corner_points) - np.array(corner_points1)
    H4pt_True = H4pt_True.flatten()
    np.save('H4pt_True.npy', H4pt_True)

    H = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(corner_points1), np.float32(perturbed_corner_points))) 
            
    img1 = cv2.warpPerspective(image, H, (w,h))
    
######## View lines #########
    img2 = cv2.warpPerspective(image, H, (w,h))
    corners_A = np.array(corner_points1)
    corners_A = corners_A.reshape((-1,1,2))
    Corners_A_transformed = cv2.perspectiveTransform(np.float32(corners_A), H)  
    Corners_B = Corners_A_transformed.astype(int)
#we got img2 and Corners_B
    img1_with_Corners = drw_cor(img2, corner_points1, (0, 0, 255))
    img2_with_Corners = drw_cor(img2, Corners_B, (0, 0, 255))


##Predicting Homography
    patch2= img1[center_y-64:center_y+64, center_x-64:center_x+64]

    patch_images=[]
    patch_images.append(patch1)
    patch_images.append(patch2)
    cv2.imwrite('patch1.jpg', patch1)
    cv2.imwrite('patch2.jpg', patch2)
    homography = predict_homographies(patch_images)
    np.save("H4pt_pred_Unsup.npy",homography)
    H4pt_pred = np.load("H4pt_pred_Unsup.npy").astype(np.float32)
#print(homography)


    cb=np.array([[homography[0]+center_x-64,homography[1]+center_y-64], [homography[2]+center_x+64, homography[3]+center_y-64] , [homography[4]+center_x-64, homography[5]+center_y+64], [homography[6]+center_x+64,homography[7]+center_y+64]]) 
    ca=np.array([[center_x-64,center_y-64], [center_x+64,center_y-64] , [center_x-64,center_y+64], [center_x+64,center_y+64]])  

    H = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(ca),np.float32(cb)))
    print(H)            


    height, width, channels = image.shape
    img2_warped = cv2.warpPerspective(image, H, (width, height))
    

##################### View lines ############################
    img2_with_Corners_pred = drw_cor(img2, cb, (0, 255, 0))  #Green

#BasePath = '../../Data/Test_sample/'
    mae = mean_absolute_error(H4pt_pred, H4pt_True)
    print("Mean absolute Error for image at index ",1, ":  ",mae)
    img2_with_Corners_pred = cv2.putText(img2_with_Corners_pred, "MCE: "+str(round(mae,3)),(150,230),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,0,0),2,cv2.LINE_AA)
    Pic_stack = np.hstack((img1_with_Corners, img2_with_Corners_pred))
    cv2.imwrite(str(1)+'.png',Pic_stack)

if __name__ == "__main__":
    main()