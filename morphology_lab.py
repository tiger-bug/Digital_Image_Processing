
from skimage.data import camera
import numpy as np
import matplotlib.pyplot as plt


def main():
    img = 1.0*camera()
    
    erosion = np.zeros((img.shape[0]-2,img.shape[1]-2))
    dialation = np.zeros((img.shape[0]-2,img.shape[1]-2))
    
    row,column = img.shape[0]-2,img.shape[1]-2

    for i in range(row):
        for j in range(column):
            
            erosion[i,j] = np.min(img[i:i+3,j:j+3])
            dialation[i,j] = np.max(img[i:i+3,j:j+3])
            
    fig,ax = plt.subplots(ncols=3, figsize=(10, 5))
    
    img_list = [('Original',img),('Erosion',erosion),('Dialation',dialation)]
    
    for z,img in enumerate(img_list):
        
        ax[z].imshow(img[1],interpolation = 'nearest',cmap=plt.cm.gray)
        ax[z].axis('off')
        ax[z].set_title(img[0])
    plt.show()
        
if __name__=='__main__':
    main()
        



    
