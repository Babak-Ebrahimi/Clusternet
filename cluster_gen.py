
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
#from tensorflow.python.tools import inspect_checkpoint as chkp
#matplotlib.use('agg')

#import tensorflow as tf
import numpy as np
import math
import itertools

import torch
import torch.nn as nn
#import torchvision
#import torchvision.transforms as transforms

import sys
import numpy as np
import glob
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
#import Image 
#from PIL import Image
import cv2
from scipy.ndimage.measurements import label
from scipy import ndimage


import matplotlib.pyplot as plt
plt.switch_backend('agg')



def read_images(file_path,extension,label):

    #images = []
    i=0
    #mypath="/home/babak/Desktop/WPAFB2009/AOI01/alaki/"
    #mypath="/home/babak/Desktop/WPAFB2009/AOI01/finalCrop/"
    mypath=file_path
    print ("mypath",mypath)
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    onlyfiles.sort()
    #print ("onlyfiles",onlyfiles)
    #images = np.empty(len(onlyfiles), dtype=object)
    images = []
    #for img in glob.glob("/home/babak/Desktop/WPAFB2009/AOI01/alaki/*.pgm"):
    #for img in glob.glob("/home/babak/Desktop/WPAFB2009/AOI01/finalCrop/*.pgm"):
    #file_index=10001
    #for img in glob.glob(mypath+"*."+extension):
    for file_index in range (10100,11125):#11026):
        if i%100==0:
            print ("i+1",i+1)
        #image2 = Image.open(file)
        #infile = open(img,'rb')
        if (extension=="pgm"):
          #img = preprocessing.minmax_scale(cv2.imread(file_path+"cropImage"+str(file_index)+".pgm",0), feature_range=(-1, 1))
          img = cv2.imread(file_path+"cropImage"+str(file_index)+".pgm",0)
          print ("img.shape",img.shape) 
            #infile = open(file_path+"cropImage"+str(file_index)+".pgm",'rb')
        if (extension=="pbm"):
          #img = preprocessing.minmax_scale(cv2.imread(file_path+"GT_cropImage"+str(file_index)+".pbm",0), feature_range=(0, 1))
          img = cv2.imread(file_path+"GT_cropImage"+str(file_index)+".pbm",0)
          print ("img.shape",img.shape) 
            #infile = open(file_path+"GT_cropImage"+str(file_index)+".pbm",'rb')
        
        images.append(img)
        
        #print ("img.shape",img.shape)  

        #images.append(image)
        i=i+1
    images=np.array(images)    

    #print ("images.shape",images.shape)  
    #print ("len(images)",len(images))
    #print ("images[0].shape",images[0].shape)
    #print ("images[0].shape",images[0].shape)
    #print ("images=\n",images)
    #print ("images.shape",*images.shape)

    #image3=tf.convert_to_tensor(images, dtype=tf.uint8)

    return images


#file = "/home/babak/Desktop/WPAFB2009/AOI34/finalCrop/"
file = "/home/babak/Desktop/WPAFB2009/AOI42/Heatmaps/BinSegHeatmapsRedacted/"

data=read_images(file,"pbm",False)
struct1 = ndimage.generate_binary_structure(2, 2)
for i in range(0,len(data)):
	temp=data[i]
	print ("temp.shape",temp.shape)
	temp2=cv2.resize(temp, dsize=(2278, 2278))
	print ("temp2.shape",temp2.shape)
	#cv2.imwrite('/home/babak/Desktop/WPAFB2009/AOI34/finalCrop2/'+"cropImage"+str(10001+i)+'.pgm',temp2)#image*255)
	for m in range(temp2.shape[0]):
		for n in range(temp2.shape[1]):
			if temp2[m,n]>127:
				temp2[m,n]=1
			else:
				temp2[m,n]=0

	structure=np.ones((3, 3),dtype=np.float64)
	labeled,ncomponents=label(temp2,structure)
	print ("ncomponents",ncomponents)
	indices = np.indices(temp2.shape).T[:,:,[1, 0]]  #np.flip(temp2.shape,0)
	#print("indices[labeled == 1]",indices[labeled == 97])
	#print (np.mean(indices[labeled == 97],axis=0))
	predicted_averages1=np.zeros((ncomponents,2),dtype = np.float64)
	for k in range(ncomponents):
		predicted_averages1[k]=np.mean(indices[labeled == k+1],axis=0)
	image=np.zeros((2278,2278))#((image_height,image_width))
	print("predicted_averages1[0]",predicted_averages1[0])
	print("predicted_averages1[0][0]",predicted_averages1[0][0])
	print("predicted_averages1[0][1]",predicted_averages1[0][1])
	print("predicted_averages1",predicted_averages1)
	for k in range(ncomponents):
		image[int(predicted_averages1[k][0]),int(predicted_averages1[k][1])]=1
	image=ndimage.binary_dilation(image, structure=struct1,iterations=20).astype(image.dtype)
	#image=ndimage.filters.gaussian_filter(image, sigma = 2, order = 0)
	image2=cv2.resize(image, dsize=(72, 72))
	cv2.imwrite('/home/babak/Desktop/WPAFB2009/AOI42/Heatmaps/BinSegClusterHeatmapsRedacted/'+"GT_cropImage"+str(10100+i)+'.pbm',image2*255)#image*255)
  	#cv2.imwrite('/home/babak/Desktop/WPAFB2009/AOI34/Heatmaps/BinSegClusterHeatmapsRedacted/'+"GT_cropImage"+str(10001+i)+'.pbm',temp2)




#BinSegClusterHeatmapsRedacted