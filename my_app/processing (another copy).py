
import cv2
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import subprocess
import os
from skimage import io
from skimage import filters, color
from scipy import ndimage as ndi
from PIL import Image
from itertools import chain
from math import ceil 
from sklearn import svm
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


plt.rcParams["figure.figsize"] = (20,6)
colors = np.array(list(chain(mcolors.BASE_COLORS.values())))        ##mcolors.CSS4_COLORS



def count_seg(filename):
    image = io.imread(filename)
    gray_image = color.rgb2gray(np.invert(image))
    thresh = filters.threshold_mean(gray_image)
    binary = gray_image > thresh
    label_arr, num_seg = ndi.label(np.invert(binary))
    return num_seg


def label_segments(filename,savename='',photo=False,marker=False):
    alpha=1.1
    beta=0
    try:
        image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    except:
        image = plt.imread(filename)
    if marker==False:
        try:
            image = cv2.blur(image, (4, 4))
        except:
            pass
    # fig,axes = plt.subplots(1,figsize=(20,10))
    # axes.imshow(image,cmap='gray')
    # axes.set_title('Raw Image')         
    if photo:
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        arrraaayyy = np.array(image)
        gray_image = color.rgb2gray(arrraaayyy)
        meaaannn = np.mean(gray_image)
        if marker:
            binary = gray_image>(meaaannn*.9)
        else:
            binary = gray_image>(meaaannn*.85)
        huh = (binary==1)*0+(binary==0)*1
        temp = savename+'_original_from_photo.png'
        io.imsave(temp,np.array(image, dtype = np.uint8 ))
        label_arr, num_seg = ndi.label(huh)   
        #plot_numbered_image(label_arr)
        segments = np.arange(1,num_seg+1)
        fig,axes = plt.subplots(1,figsize=(20,10))
        axes.imshow(huh)
        axes.set_title('Postprocessed Image')
        fig.savefig(savename+'_postprocessed.jpg')
        return huh,np.array(label_arr),segments,image #label_segments(temp,savename,photo=False)
    else:
        gray_image = color.rgb2gray(image)
        thresh = filters.threshold_mean(gray_image)
        binary = gray_image > thresh

    # io.imsave(savename+'_original.jpg',(np.array(binary*1, dtype = np.uint8 )))
#     binary = tf.image.convert_image_dtype(binary, dtype=tf.uint8)  ##suppresses warning of lossy conversion
#     io.imsave(savename+'_original.png',binary)
    label_arr, num_seg = ndi.label(np.invert(binary))
    
    segments = np.arange(1,num_seg+1)
    #print(segments)
    return binary*1,np.array(label_arr),segments,image


def plot_image(label_arr):
    plt.subplots(ncols=1, nrows=1, figsize=(8, 8))
    plt.imshow(label_arr, cmap=plt.cm.gray)
    plt.title("Labeled image")
    plt.show()


def plot_numbered_image(label_arr,savename='',no_rotate=False):
    colors = np.array(list(chain(mcolors.TABLEAU_COLORS.values())))
    # np.repeat(colors,2)                                         ### put in repeat for large sets
    pixarray=np.rot90(label_arr,3)
    imax,jmax = pixarray.shape
    fig,ax=plt.subplots(ncols=1, nrows=1, figsize=(20,int(20*jmax/imax)))
    plt.xticks(np.arange(0,imax))
    plt.yticks(np.arange(0,jmax))
    np.random.shuffle(colors)
    for i in range(imax):
        for j in range(jmax):
            val = pixarray[i][j]
            if val != 0:
                ax.text(i,j,val,fontsize=20,color=colors[val])
    plt.xticks([])
    plt.yticks([])            
    plt.show()
    fig.savefig(savename+'_segmented.png')
