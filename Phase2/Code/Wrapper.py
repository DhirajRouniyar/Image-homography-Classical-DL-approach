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

import numpy as np
import os 
import cv2
import csv
import tensorflow as tf
import pandas as pd
# Add any python libraries here


def Create_Patch(image):
    
    h, w = image.shape[:2]
    
    psi = 32
    patch_sz = 128
    
    img_right = patch_sz + 42 
    
    x = np.random.randint(42, w - img_right) 
    y = np.random.randint(42, h - img_right)
    pa = [[x, y], [x, patch_sz + y] , [patch_sz + x, y], \
                         [patch_sz + x, patch_sz + y]]
    pa = np.array(pa)
    #Create random patch P_b
    pb = np.zeros_like(pa)

    for i, points in enumerate(pa):
        pb[i][0] = points[0] + np.random.randint(- psi, psi)
        pb[i][1] = points[1] + np.random.randint(- psi, psi)
        
    inv_homography = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(pa), np.float32(pb))) 

    img_b = cv2.warpPerspective(image, inv_homography, (w,h))

    patch_PA = image[y : y + patch_sz, x : x + patch_sz]
    patch_PB = img_b[y : y + patch_sz, x : x + patch_sz]
    H4_pts = (pb - pa) 
    H4_pts = H4_pts.astype(np.float32)
    pa_pb = np.dstack((pa, pb))
    return patch_PA, patch_PB, H4_pts, pa_pb, pa, [pa, pb]
    



def main():
        
    get_dir = '../../Data/Train/'
    save_dir = '../../Data/Trained_/'    #Make this directory manually Trained_ or Validated_
    #get_dir = '../Data/Val/'            #Remove hatch# to use the validation data
    #save_dir = '../Data/Validated_/'    #Remove hatch# to save the validation data patches
    total = os.listdir(get_dir)
    no_images = len(total) + 1
    H4_pts_list = []
    img_lst = [] 
    pts_lst = []
    points_CA_list = []
    
    for image in range(1, no_images):

        a_image = cv2.imread(get_dir + str(image) + '.jpg')
        a_image = cv2.resize(a_image, (320,240), interpolation = cv2.INTER_AREA)

        pA, pB, h_4_pts, points, points_CA, _ = Create_Patch(a_image)      

        cv2.imwrite(save_dir + 'Patch_A/' + str(image) + '_' + '.jpg', pA)
        cv2.imwrite(save_dir + 'Patch_B/' + str(image) + '_' + '.jpg', pB)
        cv2.imwrite(save_dir + 'Image_IA/' + str(image) + '_' + '.jpg', a_image)           
                    
        H4_pts_list.append(h_4_pts)
        pts_lst.append(points)
        points_CA_list.append(points_CA)
        img_lst.append(str(image) + '_' + '.jpg')
                      
    homography = np.array(H4_pts_list)
    print("Homography: ", homography.shape)
    #print("Homography shape: ", homography.shape)
    np.save(save_dir + 'h4_list.npy', homography)
        
    CA_corners = np.array(points_CA_list)
    print("CA_corners: ", CA_corners.shape)
    #print("Ca_corners shape: ", CA_corners.shape)
    np.save(save_dir + 'pts_CA_list.npy', CA_corners)
        
    np.save(save_dir + "pts_lst", np.array(pts_lst))
    print("\nPoints data saved @:  ", save_dir)    
            
            
            
if __name__ == '__main__':
    main()
 