# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:06:18 2019

@author: sspiegel
"""

import os
from skimage.io import imread
from glob import glob
from skimage.exposure import histogram
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
path = r'D:\Digital_Image_Processing\Lab_7\Lab_7'
os.chdir(path)
#Load in images
img_list = glob('*.png')
#Problem 1
#print('Figures for problem 1')
for txt in img_list:
    
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    img = imread(txt,as_gray=True)
    hist,hist_centers = histogram(img)
    ax[0].imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax[0].axis('off')
    
    ax[1].plot(hist_centers, hist, lw=2)
    ax[1].set_title('Histogram of grey values')

## Problem 2
#print('Figures for problem 2')

fig,ax = plt.subplots(ncols=len(img_list), figsize=(10, 5))
for z,txt in enumerate(img_list):
    img = imread(txt,as_gray=True)
    seg = np.zeros(img.shape)
    avg = np.average(img)
    row, column = img.shape
    for i in range(row):
        for j in range(column):
            if img[i,j]>avg:
                seg[i,j]=1.0
    ax[z].imshow(seg, cmap=plt.cm.gray)
    ax[z].axis('off')
    ax[z].set_title('Global '+str(z+1))
    
#Problem 3
#print('Figures for problem 3')
fig,ax = plt.subplots(ncols=len(img_list), figsize=(10, 5))

for z,txt in enumerate(img_list):
#    txt = img_list[0]
    img = imread(txt,as_gray=True)
    thresh = threshold_otsu(img)
    binary = img <= thresh
    ax[z].imshow(binary,interpolation='nearest',cmap=plt.cm.gray)
    ax[z].axis('off')
    ax[z].set_title('Otsu '+str(z+1))
#    plt.show()
    
    
    


    

    
            
    
