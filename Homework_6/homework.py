#import modules
import numpy as np
#from skimage import io
import os
import matplotlib.pyplot as plt
from skimage import data
#set path to homework
path = r'D:\Digital_Image_Processing\Homework_6'
os.chdir(path)
####################get that face!
#img_path = 'face_3.jpg'
####################read image!
#img = io.imread(img_path,as_gray='True')
img = data.camera()
img = img.astype(float)
####################get a place holder for the LBP
new_img = np.zeros(img.shape)
####################create matrix by which to multiply
mult_mat = np.power(2,np.array([[0,1,2],[7,0,3],[6,5,4]]))
####################pad original with zeroes
img_pand = np.pad(img,1,'constant')
####################get correct row and column stuff
row,column = np.array(range(1,img.shape[0]+1)),np.array(range(1,img.shape[1]+1))
####################loop through and compute!
for i in row:
    for j in column:
        a = img_pand[i-1:i+2,j-1:j+2]
        center = a[1,1]
        a - center
        test = img_pand[i-1:i+2,j-1:j+2]-center>=0
        test = test.astype(int)
        test[1,1]=0   
        sum_array = test*mult_mat
        answer = np.sum(sum_array) 
        ####################get the appropriate corresponding pixel in new image
        new_img[i-1,j-1]=answer
        
####################plot the faces.  
fig,ax = plt.subplots(ncols = 2,figsize = (10,5))
ax[0].imshow(img,cmap = 'gray')
ax[0].set_title('Original')
ax[0].axis('off')
ax[1].imshow(new_img,cmap = 'gray')
ax[1].set_title('Look at that hansome face!')
ax[1].axis('off')

 
        