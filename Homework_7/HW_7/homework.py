# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 08:55:38 2019

@author: sspiegel
"""

from sklearn.cluster import KMeans
import os
import numpy as np
from glob import glob
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.segmentation import random_walker


path = r'D:\Digital_Image_Processing\Homework_7\HW_7'
os.chdir(path)

img_list = glob('*png')
img_list
random_seed = 0

img_ = img_list


#Problem1
fig,ax = plt.subplots(ncols=len(img_list), figsize=(10, 5))

for z,img_ in enumerate(img_list):
    
    img = imread(img_,as_gray = True)
    hist,centers = histogram(img)
    ax[z].plot(centers,hist)
    ax[z].set_title('Image '+str(z))

plt.tight_layout()   
    #######################################################################
###Kmeans
labels_list = []
for z,img_ in enumerate(img_list):
    img = imread(img_,as_gray=True)
    img = gaussian(img)
    flat = img.flatten()
    kmeans = KMeans(n_clusters=2,random_state=random_seed).fit(flat[:,None])

    labels = kmeans.labels_
    labels_img = np.reshape(labels,img.shape)
    labels_list.append(labels_img)

fig,ax = plt.subplots(ncols=len(labels_list), figsize=(10, 5))
for z,label in enumerate(labels_list):
    ax[z].imshow(label,cmap = 'gray')
    ax[z].axis('off')
    ax[z].set_title('Kmeans '+str(z))



####Otsu######################
binary_list = []
for z,img_ in enumerate(img_list):
    img = imread(img_,as_gray=True)
    img = gaussian(img)
    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary_list.append(binary)
    

fig,ax = plt.subplots(ncols=len(labels_list), figsize=(10, 5))
for z,label in enumerate(binary_list):
    ax[z].imshow(label,cmap = 'gray')
    ax[z].axis('off')
    ax[z].set_title('Otsu '+str(z))
 


for z,img_ in enumerate(img_list):
    data = 1.0*imread(img_,as_gray=True)
    #data = rescale_intensity(data)
    data = gaussian(data,sigma=1.0)
    #data = adjust_sigmoid(data)
    plt.imshow(data,cmap='gray')
    
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data < np.average(data)-np.std(data)] = 1
    markers[data > np.average(data)+np.std(data)] = 2
    
    # Run random walker algorithm
    labels = random_walker(data, markers,beta = 150, mode='bf')
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                        sharex=True, sharey=True)
    ax1.imshow(data, cmap='gray', interpolation='nearest')
    ax1.axis('off')
    ax1.set_title( 'Image '+str(z)+' data')
    ax2.imshow(markers, cmap='magma', interpolation='nearest')
    ax2.axis('off')
    ax2.set_title('Image '+str(z)+' Markers')
    ax3.imshow(labels, cmap='gray', interpolation='nearest')
    ax3.axis('off')
    ax3.set_title('Image '+str(z)+' Segmentation')
    
    fig.tight_layout()
    plt.show()

'''The segmentation worked really well with Kmeans and Otsu segmentation on the non noisy and 
noisy images.  The region growing worked well with the nonnoisy data but did not work with the noisy data, even when I denoised it with a gaussian filter.  Perhaps this was with limitations with random walker segmentation or with my parameters.  Perhaps an intensity transform after the gaussian filter would have helped as well.  '''

