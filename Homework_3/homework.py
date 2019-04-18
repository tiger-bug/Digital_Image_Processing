
'''Import necessary libraries'''

from skimage import io
from skimage.exposure import histogram
from skimage.filters import gaussian
#import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import roberts, sobel, prewitt
import os
path = r'D:\Digital_Image_Processing\Homework_3'

'''Get the path to the image'''
os.chdir(path)

img_txt = os.listdir()[0]


'''Problem 1'''


'''Read the image'''
image = io.imread(img_txt,as_gray=True)
'''compute the histogram of the grayscale image, using local neighborhood of 
each pixel'''
plt.imshow(image,cmap = 'gray')
hist, centers = histogram(image,nbins=256)
#plt.hist(hist, bins = centers.size)
'''Juxtapose the image with its corresponding histrgram'''
fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(image, interpolation='nearest', cmap=plt.cm.gray)
ax[0].axis('off')
ax[1].plot(centers, hist, lw=2)
ax[1].set_title('Histogram of grey values')

#####################################################################################3

'''Problem 2'''
n_cols = 4
cmap = 'gray'
edge_sobel = sobel(image)
edge_roberts = roberts(image)
edge_prewitt = prewitt(image)
img_list = [(image,'Original'),(edge_sobel,'Sobel filter'),(edge_roberts,'Roberts filter'),(edge_prewitt,'Prewitt filter')]
#edge_sobel = np.reshape(edge_sobel,image.shape)
fig,ax = plt.subplots(ncols = n_cols, figsize=(14,7))
for i,im in enumerate(img_list):
    ax[i].imshow(im[0],cmap=cmap)
    ax[i].axis('off')
    ax[i].set_title(im[1])
    
##############################################################################################

'''Problem 3'''
fig,ax = plt.subplots(ncols = n_cols, figsize=(14,7))
sigma_list = [0.3,0.8,1.0]
ax[0].imshow(image,cmap = cmap)
ax[0].axis('off')
ax[0].set_title('Original')
for i,im in enumerate(sigma_list):
    gauss = gaussian(image,sigma = im)
    ax[i+1].imshow(gauss,cmap = cmap)
    ax[i+1].axis('off')
    ax[i+1].set_title('Sigma = '+str(im))


