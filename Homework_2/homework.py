
import os
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_sauvola)





matplotlib.rcParams['font.size'] = 9
#Path to homework directory
path = r'D:\Digital_Image_Processing\Homework_2'
#Change directory to my homework directory
os.chdir(path)
#Read the image as greyscale
image = io.imread(os.listdir()[0],as_gray = True)


'''Set Otsu threshold as a boolean array.  The pixels that have a value
above the global threshold.
'''
binary_global = image > threshold_otsu(image)

"""Set the window size for the Niblack and Sauvola threshold."""

window_size = 25
thresh_niblack = threshold_niblack(image, window_size=window_size, k=0.8)
thresh_sauvola = threshold_sauvola(image, window_size=window_size)



binary_niblack = image > thresh_niblack
binary_sauvola = image > thresh_sauvola

"""Plot the original greyscale, global, niblack, and Sauvola.  True values
in the array will plot as white and false values will plot as black."""

plt.figure(figsize=(8, 7))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap=plt.cm.gray)
plt.title('Original (as greyscale)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Global Threshold')
plt.imshow(binary_global, cmap=plt.cm.gray)
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(binary_niblack, cmap=plt.cm.gray)
plt.title('Niblack Threshold')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(binary_sauvola, cmap=plt.cm.gray)
plt.title('Sauvola Threshold')
plt.axis('off')

plt.show()
