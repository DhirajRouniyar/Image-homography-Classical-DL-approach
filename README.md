# Image-homography-Classical-DL-approach

# PROBLEM STATEMENT:  
The purpose of the project is to stitch two or more images to create a panorama image. The images should have repeated
local features which can be used to match features and blend images together. Two approaches are considered in the project  
i.e. A traditional approach and a Deep Learning approach.
# PHASE1: Traditional approach   
This section focuses on the implementation of the algorithm to generate a panorama out of multiple images with repeated
local features. Two image sets need to be captured that have about 30 to 50 percent image overlap between them. This
approach has the following steps:    

# PHASE2: Supervised and Unsupervised approach   
This segment, we applied both Supervised and Unsupervised deep learning techniques to calculate the 4-point
Homography (H4P t) matrix between two images for the
purpose of overlaying image with homography estimated by
deep learning model and ground truth.  

Refer ![Report](https://drive.google.com/file/d/12P34wbqmIqM71LsVZH9jaAxe_pgtopfu/view?usp=sharing) for details.  

# SOME RESULTS:  

• Corner Detection  
• Adaptive Non-Maximal Suppression (ANMS)  
• Feature Descriptor  
• Feature Matching  
• RANSAC for outlier rejection and to estimate Robust Homography  
• Blending Images  

# Input Images
![Alt Text](https://github.com/DhirajRouniyar/Assets/blob/main/Images/Input%20images.png)
# Panaroma Stiching
![Alt Text](https://github.com/DhirajRouniyar/Assets/blob/main/Images/Panaroma.png)
