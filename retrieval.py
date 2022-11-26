import os
import cv2
import numpy as np
import glob
import tensorflow as tf
%matplotlib inline
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from google.colab import drive
import torchvision.datasets as datasets
drive.mount('/content/drive') 
path = "/content/drive/My Drive/Datasets/VOC2007" 

os.chdir(path)
os.listdir(path)

annotation_path = path+"/VOCdevkit/VOC2007/ImageSets/Main/"
print(os.listdir(annotation_path))

image_path = path+"/VOCdevkit/VOC2007/JPEGImages/"
print(os.listdir(image_path))

# Putting all of the images in one dataset
Images = []
for i in glob.glob('/content/drive/My Drive/Datasets/VOC2007/VOCdevkit/VOC2007/JPEGImages/*.jpg'):
    img = cv2.imread(i)
    Images.append(img)

print(len(Images))

# Picking an image from the dataset
Image1 = Images[777]
plt.imshow(Image1)
plt.show

#creating the sift function
sift = cv2.xfeatures2d.SIFT_create()

from tqdm.auto import tqdm
kpI, desI = sift.detectAndCompute(np.array(Image1),None)
best_match = []
best_img = []
best_kp = []
best_matches = []
for i in tqdm(range(650,800)):
    img2 = Images[i]
    
    kp2, des2 = sift.detectAndCompute(np.array(img2),None)
 
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desI,des2, k=2)

    good = []

    for m,n in matches:
      if m.distance < 0.90*n.distance:  # threshold can be changed
        good.append([m])


    if len(good) >= len(best_match):
      best_match = good
      best_img = img2
      best_kp = kp2
      Fimg = cv2.drawMatchesKnn(np.array(Image1),kpI,np.array(best_img),best_kp,best_match,None,flags=2)
      plt.imshow(np.array(Fimg))
      plt.show()