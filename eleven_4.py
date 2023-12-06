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
import torchvision.transforms as transforms


import sys
import numpy as np
import glob
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing
from skimage import transform
#import Image 
#from PIL import Image
import cv2
from scipy.ndimage.measurements import label

import matplotlib.pyplot as plt
plt.switch_backend('agg')


#from keras.models import Sequential
#from keras.layers import InputLayer, Input
#from keras.layers import Reshape, MaxPooling2D
#from keras.layers import Conv2D, Dense, Flatten
import math
#from six.moves import input
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=6)
torch.set_printoptions(precision=6)




class ConvNet(nn.Module):
    def __init__(self,num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential( 
            nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.PReLU(num_parameters=1, init=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(num_parameters=1, init=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(num_parameters=1, init=0.001))
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.PReLU(num_parameters=1, init=0.001))
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(num_parameters=1, init=0.001),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(num_parameters=1, init=0.001))
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(1),
            #nn.PReLU(num_parameters=1, init=0.001),
            nn.Sigmoid())
            #nn.Sigmoid())
        #self.fc = nn.Linear(7*7*32, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        #print ("1: ",out.size())
        out = self.layer2(out)
        #print ("2: ",out.size())
        out = self.layer3(out)
        #print ("3: ",out.size())
        out = self.layer4(out)
        #print ("4: ",out.size())
        out = self.layer5(out)
        #print ("5: ",out.size())
        out = self.layer6(out)
        #print ("6: ",out.size())
        out = self.layer7(out)
        #print ("7: ",out.size())
        #out = out.reshape(out.size(0), -1)
        #out = self.fc(out)
        return out.squeeze()
    '''
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    '''

class FoveaNet(nn.Module):
    def __init__(self,num_classes=2):
        super(FoveaNet, self).__init__()
        self.layer1 = nn.Sequential( 
            nn.Conv2d(in_channels=5, out_channels=16, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.AdaptiveMaxPool2d((261,261)))
            nn.MaxPool2d(kernel_size=2, stride=2))
        #self.layer2 = nn.Sequential(
        #    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=15, stride=1, padding=0),
        #    nn.BatchNorm2d(16),
        #    nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=13, stride=1, padding=6),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(256),
            nn.ReLU())
            #nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU())
            #nn.Dropout(p=0.5))
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
            #nn.Dropout(p=0.5))
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0),
            #nn.BatchNorm2d(1),
            #nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Sigmoid())
        #self.fc = nn.Linear(7*7*32, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        #print ("1: ",out.size())
        out = self.layer2(out)
        #print ("2: ",out.size())
        out = self.layer3(out)
        #print ("3: ",out.size())
        out = self.layer4(out)
        #print ("4: ",out.size())
        out = self.layer5(out)
        #print ("5: ",out.size())
        out = self.layer6(out)
        #print ("6: ",out.size())
        out = self.layer7(out)
        #print ("7: ",out.size())
        out = self.layer8(out)
        #print ("8: ",out.size())
        #out = self.layer9(out)
        #print ("9: ",out.size())
        #out = out.reshape(out.size(0), -1)
        #out = self.fc(out)
        return out.squeeze()


def read_images(file_path,extension,istest_data):

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
    #for file_index in range (10001,11026):#11026):

    for file_index in range (0,len(onlyfiles)):#100):#2049):
        if i%100==0:
            print ("i+1",i+1)
        #image2 = Image.open(file)
        #infile = open(img,'rb')
        if (extension=="pgm" and istest_data==False):
          #img = preprocessing.minmax_scale(cv2.imread(file_path+"cropImage"+str(file_index)+".pgm",0), feature_range=(-1, 1))
          img = cv2.imread(file_path+"cropImage"+str(10000+file_index)+".pgm",0)
            #infile = open(file_path+"cropImage"+str(file_index)+".pgm",'rb')
        if (extension=="pbm" and istest_data==False):
          #img = preprocessing.minmax_scale(cv2.imread(file_path+"GT_cropImage"+str(file_index)+".pbm",0), feature_range=(0, 1))
          img = cv2.imread(file_path+"GT_cropImage"+str(10000+file_index)+".pbm",0)
            #infile = open(file_path+"GT_cropImage"+str(file_index)+".pbm",'rb')
        if (extension=="pgm" and istest_data==True):
          #img = preprocessing.minmax_scale(cv2.imread(file_path+"cropImage"+str(file_index)+".pgm",0), feature_range=(-1, 1))
          img = cv2.imread(file_path+"cropImage"+str(10001+file_index)+".pgm",0)
            #infile = open(file_path+"cropImage"+str(file_index)+".pgm",'rb')
        if (extension=="pbm" and istest_data==True):
          #img = preprocessing.minmax_scale(cv2.imread(file_path+"GT_cropImage"+str(file_index)+".pbm",0), feature_range=(0, 1))
          img = cv2.imread(file_path+"GT_cropImage"+str(10001+file_index)+".pbm",0)
            #infile = open(file_path+"GT_cropImage"+str(file_index)+".pbm",'rb')
        
        images.append(img)


        #images.append(image)
        i=i+1
    images=np.array(images)    

    #print ("images1.shape",images.shape)  
    #print ("len(images)",len(images))
    #print ("images[0].shape",images[0].shape)
    #print ("images[0].shape",images[0].shape)
    #print ("images=\n",images)
    #print ("images.shape",*images.shape)

    #image3=tf.convert_to_tensor(images, dtype=tf.uint8)

    return images

def read_images_2(file_path,extension,istest_data):
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
  #for file_index in range (10001,11026):#11026):
  for file_index in range (0,len(onlyfiles)):#100):#2049):
    if i%100==0:
      print ("i+1",i+1)
      #image2 = Image.open(file)
      #infile = open(img,'rb')
    if istest_data==False :  
    #img = preprocessing.minmax_scale(cv2.imread(file_path+"GT_cropImage"+str(file_index)+".pbm",0), feature_range=(0, 1))
      img = cv2.imread(file_path+"GT_cropImage"+str(10000+file_index)+".pbm",0)
      #img=img.reshape(2278, 2278)
    if istest_data==True :  
      img = cv2.imread(file_path+"GT_cropImage"+str(10001+file_index)+".pbm",0)
    images.append(img)
    i=i+1
  images=np.array(images)
  #print ("images.shape",images.shape)
  return images


def add_frames_to_channels(data):
    #dd=np.expand_dims(d,axis=2)
    #k=np.concatenate((aa,bb,cc,dd),axis=2)
    num_channels=5
    #data_extended=np.zeros((data.shape[0],data.shape[1],data.shape[2],num_channels),np.float64)
    data_extended=np.zeros((data.shape[0],data.shape[1],data.shape[2],num_channels),np.uint8)
    data = np.expand_dims(data, axis=3)
    #print ("data.shape =",*data.shape)
    for i in range(0,data.shape[0]):
        if i%100==0:
            print ("i =",i)

        if i-2<0:
            frame_minus_2=data[i]
        else:
            frame_minus_2=data[i-2]

        #print ("frame_minus_2.shape =",*frame_minus_2.shape)
        if i-1<0:
            frame_minus_1=data[i]
        else:
            frame_minus_1=data[i-1]

        #print ("frame_minus_1.shape =",*frame_minus_1.shape)
        frame_middle=data[i]
        #print ("frame_middle.shape =",*frame_middle.shape)
        if i+1>(data.shape[0]-1):
            frame_plus_1=data[i]
        else:
            frame_plus_1=data[i+1]
        #print ("frame_plus_1.shape =",*frame_plus_1.shape)
        if i+2>(data.shape[0]-1):
            frame_plus_2=data[i]
        else:
            frame_plus_2=data[i+2]
        #print ("frame_plus_2.shape =",*frame_plus_2.shape)
        data_extended[i]=np.concatenate((frame_minus_2,frame_minus_1,frame_middle,frame_plus_1,frame_plus_2),axis=2)

    #print ("data_extended.shape =",*data_extended.shape)
    return data_extended

def add_frames_to_channels2(data,i):
    #dd=np.expand_dims(d,axis=2)
    #k=np.concatenate((aa,bb,cc,dd),axis=2)
    num_channels=5
    #data_extended=np.zeros((data.shape[0],data.shape[1],data.shape[2],num_channels),np.float64)
    #print ("data.shape-----------------",data.shape)
    data_extended=np.zeros((num_channels,data.shape[1],data.shape[2]),np.uint8)
    data = np.expand_dims(data, axis=1)
    #print ("***********data_extended.shape",data_extended.shape)

    #print ("data[0].shape-----------------",data[0].shape)
    #print ("data.shape =",*data.shape)
    #for i in range(0,data.shape[0]):
    #    if i%100==0:
    #        print ("i =",i)

    if i-2<0:
      frame_minus_2=data[i]
    else:
      frame_minus_2=data[i-2]

        #print ("frame_minus_2.shape =",*frame_minus_2.shape)
    if i-1<0:
      frame_minus_1=data[i]
    else:
      frame_minus_1=data[i-1]

        #print ("frame_minus_1.shape =",*frame_minus_1.shape)
    frame_middle=data[i]
        #print ("frame_middle.shape =",*frame_middle.shape)
    if i+1>(data.shape[0]-1):
      frame_plus_1=data[i]
    else:
      frame_plus_1=data[i+1]
        #print ("frame_plus_1.shape =",*frame_plus_1.shape)
    if i+2>(data.shape[0]-1):
      frame_plus_2=data[i]
    else:
      frame_plus_2=data[i+2]
        #print ("frame_plus_2.shape =",*frame_plus_2.shape)
    #data_extended[i]=np.transpose(np.concatenate((frame_minus_2,frame_minus_1,frame_middle,frame_plus_1,frame_plus_2),axis=2),(0,3,1,2))
    data_extended=np.concatenate((frame_minus_2,frame_minus_1,frame_middle,frame_plus_1,frame_plus_2),axis=0)
    #print ("-----------------------data_extended.shape",data_extended.shape)

    #print ("data_extended.shape =",*data_extended.shape)
    return data_extended




def outFromIn(conv, layerIn):
  n_in = layerIn[0]
  j_in = layerIn[1]
  r_in = layerIn[2]
  start_in = layerIn[3]
  k = conv[0]
  s = conv[1]
  p = conv[2]
  
  n_out = math.floor((n_in - k + 2*p)/s) + 1
  actualP = (n_out-1)*s - n_in + k 
  pR = math.ceil(actualP/2)
  pL = math.floor(actualP/2)
  
  j_out = j_in * s
  r_out = r_in + (k - 1)*j_in
  start_out = start_in + ((k-1)/2 - pL)*j_in
  return n_out, j_out, r_out, start_out
  
def printLayer(layer, layer_name):
  print(layer_name + ":")
  print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))



def helper(x,y,convnet,layer_names,imsize,layer):
  #convnet =   [[3,2,2],[2,2,0],[3,2,2],[2,2,0],[3,1,2],[1,1,0],[3,1,1],[2,2,0],[3,1,1], [1, 1, 0]]#,[6,1,0], [1, 1, 0]]
  #layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','conv6', 'conv7']#,'fc6-conv', 'fc7-conv']
  #imsize = 2278

  #convnet =   [[15,1,0],[2,2,0],[15,1,0],[13,1,0],[11,1,0],[9,1,0],[7,1,0],[5,1,0],[3,1,0], [1, 1, 0]]
  #layer_names = ['conv0','pool1','conv1','conv2','conv3','conv4','conv5','conv6','conv7', 'conv8']
  #imsize=261
 
  layerInfos = []
  #first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
  #print ("-------Net summary------")
  currentLayer = [imsize, 1, 1, 0.5]
  #printLayer(currentLayer, "input image")
  for i in range(len(convnet)):
    currentLayer = outFromIn(convnet[i], currentLayer)
    layerInfos.append(currentLayer)
    #88888888888888888888888888888888888printLayer(currentLayer, layer_names[i])
  #print ("------------------------")
  layer_name =layer#'conv7' #****************input ("Layer name where the feature in: ")
  #print('layer_names',layer_names)
  layer_idx = layer_names.index(layer_name)
  idx_x = int(x)#************input ("index of the feature in x dimension (from 0)"))
  idx_y = int(y)#************input ("index of the feature in y dimension (from 0)"))
  
  n = layerInfos[layer_idx][0]
  j = layerInfos[layer_idx][1]
  r = layerInfos[layer_idx][2]
  start = layerInfos[layer_idx][3]
  #print ("idx_x=",idx_x)
  #print ("idx_y=",idx_y)
  #print ("n=",n)
  assert(idx_x < n)
  assert(idx_y < n)
  
  #print ("receptive field: (%s, %s)" % (r, r))
  #print ("center: (%s, %s)" % (start+idx_x*j, start+idx_y*j))
  return r,start+idx_x*j,start+idx_y*j

def make_train_data_fovea(tr_data,tr_label,tr2_label,k):
  convnet =   [[3,2,2],[2,2,0],[3,2,2],[2,2,0],[3,1,2],[1,1,0],[3,1,1],[2,2,0],[3,1,1], [1, 1, 0]]#,[6,1,0], [1, 1, 0]]
  layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','conv6', 'conv7']#,'fc6-conv', 'fc7-conv']


  #Fovnet =   [[15,1,7],[2,2,0],[13,1,6],[11,1,5],[9,1,4],[7,1,3],[5,1,2],[3,1,1], [1, 1, 0]]
  #layer_names = ['conv0','pool1','conv1','conv2','conv3','conv4','conv5','conv6','conv7']
 
  imsize2=2278
  #imsize2=2340
  imsize=261
  layer='conv7'
  #crop_size=67
  cr_size=4
  #crop_size=130
  new_tr_label=[]
  new_tr_data=[]
  point_list=[]
  #print("tr_data.shape",tr_data.shape)
  print ("-----------------------------------------")
  print("tr_data.shape",tr_data.shape)
  print("tr2_label.shape",tr2_label.shape)
  x_step=int(tr_label.shape[0]/cr_size)
  y_step=int(tr_label.shape[1]/cr_size)
  print ("x_step",x_step)
  print ("y_step",y_step)
  counter=0;
  #print ("tr2_label=",tr2_label)
  for i in range (0,x_step):
    for j in range (0,y_step):
      #print("i*crop_size",i*crop_size)
      #print("(i+1)*(crop_size)",(i+1)*(crop_size))
      #print("j*crop_size",j*crop_size)
      #print("(j+1)*(crop_size)",(j+1)*(crop_size))
      check_nonzero=tr_label[i*cr_size:(i+1)*(cr_size),j*cr_size:(j+1)*(cr_size)]
      #print("check_nonzero.shape",check_nonzero.shape)
      if np.count_nonzero(check_nonzero)>0:
        #print("\ndoooooooooooooooooooon't worry")
        '''
        find_car_flag=False
        for k  in range (0,67):
          for l in range (0,67):
            if tr2_label[i*crop_size+k][j*crop_size+l]==1:
              find_car_flag=True
              break
          if find_car_flag==True:
            break

        if find_car_flag==True:
        '''
        #for m  in range (0,67):
          #for n in range (0,67):
        #temp=np.zeros((crop_size,crop_size))
        #important############################temp=tr2_label[i*crop_size:(i+1)*crop_size,j*crop_size:(j+1)*crop_size]
        #print ("temp.shape",temp.shape)
        #importnatn###########################new_tr_label.append(temp)#***********************
        
        point_list.append([i,j])#*******************
        rec_size,center1_x,center1_y=helper(i*cr_size,j*cr_size,convnet,layer_names,imsize2,layer)
        rec_size,center2_x,center2_y=helper((i+1)*cr_size-1,(j+1)*cr_size-1,convnet,layer_names,imsize2,layer)
        rec_size=165
        center1_x=center1_x#i*crop_size
        center1_y=center1_y#j*crop_size
        center2_x=center2_x#(i+1)*crop_size-1
        center2_y=center2_y#(j+1)*crop_size-1
        temp=np.zeros((imsize,imsize))
        temp2=(-1.0)*np.ones((5,imsize,imsize))
        pad_value=rec_size/2#(imsize-crop_size)//2
        #print ("tr_data.shape",tr_data.shape)
        #if i==0 and j==0:
        #  print("tr_data[0]",tr_data[2,0:10,0:10])
        #print ("temp2.shape",temp2.shape)
        #print ("tr2_label.shape",tr2_label.shape)
        #print ("1-----",i*crop_size-pad_value)
        #print ("2-----",i*crop_size+crop_size+pad_value)
        #print ("3-----",j*crop_size-pad_value)
        #print ("4-----",j*crop_size+crop_size+pad_value)
        #for m in range (i*crop_size-pad_value,i*crop_size+crop_size+pad_value):
         # for n in range (j*crop_size-pad_value,j*crop_size+crop_size+pad_value):
          #  if m>=0 and n>=0 and m<2278 and n<2278 :#tr2_label.shape[0] and n<tr2_label.shape[1]:
           #   temp2[:,m-(i*crop_size-pad_value),n-(j*crop_size-pad_value)]=tr_data[:,m,n]
              #temp2[:,m,n]=tr_data[:,m-(i*crop_size-pad_value),n-(j*crop_size-pad_value)]
              #print ("temp2=",temp2)
        #print("center1_x",center1_x)
        #print("center1_y",center1_y)
        #print("center2_x",center2_x)
        #print("center2_y",center2_y)
        #print("pad_value",pad_value)
        #print("tr_data.shape",tr_data.shape)
        #print("m-(center1_x-pad_value)",m-(center1_x-pad_value))
        #print("n-(center1_y-pad_value)",n-(center1_y-pad_value))
        for m in range (int(center1_x-pad_value),int(center2_x+pad_value)):
          for n in range (int(center1_y-pad_value),int(center2_y+pad_value)):
            if m>=0 and n>=0 and m<2278 and n<2278:#tr2_label.shape[0] and n<tr2_label.shape[1]:
              temp[m-int(center1_x-pad_value),n-int(center1_y-pad_value)]=tr2_label[m,n]
              temp2[:,m-int(center1_x-pad_value),n-int(center1_y-pad_value)]=tr_data[:,m,n]
              
        new_tr_label.append(cv2.resize(temp, dsize=(130, 130)))
        new_tr_data.append(temp2)#***************************************
        counter+=1
  print("counter=",counter)
  #print("new_tr_data.shape",new_tr_data.shape)
  #print("new_tr_label.shape",new_tr_label.shape)

  return new_tr_data,new_tr_label,point_list




def make_test_data_fovea(ts_data,ts_label,predicted,k):
  print ("test_data.shape",ts_data.shape)
  print ("test_label.shape",ts_label.shape)
  #print ("test_data.size",ts_data.size)
  print ("test_label.size",ts_label.size)
  convnet =   [[3,2,2],[2,2,0],[3,2,2],[2,2,0],[3,1,2],[1,1,0],[3,1,1],[2,2,0],[3,1,1], [1, 1, 0]]#,[6,1,0], [1, 1, 0]]
  #Fovnet =   [[15,1,0],[2,2,0],[15,1,0],[13,1,0],[11,1,0],[9,1,0],[7,1,0],[5,1,0],[3,1,0], [1, 1, 0]]
  #layer_names = ['conv0','pool1','conv1','conv2','conv3','conv4','conv5','conv6','conv7', 'conv8']
  layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool3','conv6', 'conv7']
  imsize=2278
  #imsize=261
  layer='conv7'
  #crop_size=4
  #crop_size=130
  chip_size=4
  new_ts_label=[]
  new_ts_data=[]
  point_list=[]
  x_step=int(predicted.shape[0]/chip_size)#crop_size)
  y_step=int(predicted.shape[1]/chip_size)#crop_size)
  for i in range (0,x_step):
    for j in range (0,y_step):


      check_nonzero=torch.from_numpy(predicted[i*chip_size:(i+1)*(chip_size),j*chip_size:(j+1)*(chip_size)]).type(torch.FloatTensor).to(device1)
      #print("check_nonzero.shape",check_nonzero.shape)
      if torch.nonzero(check_nonzero).size(0)>0 :
      #if torch.sum(check_nonzero>0.5)>0:
        #print ("torch.sum(check_nonzero!=0)",torch.sum(check_nonzero!=0))
        #print ("check_nonzero",check_nonzero)
      #find_car_flag=False
      #for k  in range (0,4):
      #  for l in range (0,67):
      #    if train2_label[i*crop_size+k][j*crop_size+l]==1:
      #      find_car_flag=True
      #      break
      #  if find_car_flag==True:
      #    break

      #if find_car_flag==True:
        #for m  in range (0,67):
          #for n in range (0,67):
        #rec_size,ceter1_x,center1_y=helper(i*crop_size,j*crop_size,convet,layer_names,imsize,layer)
        #rec_size,ceter2_x,center2_y=helper(i*crop_size+crop_size-1,j*crop_size+crop_size-1,Fovnet,layer_names,imsize,layer)
        #################temp=test_label[i*crop_size:(i+1)*(crop_size),j*crop_size:(j+1)*(crop_size)]
        ###############new_test_label.append(temp)#***********************
        ##############point_list.append([i*crop_size,j*crop_size])#*******************
        rec_size,center1_x,center1_y=helper(i*chip_size,j*chip_size,convnet,layer_names,imsize,layer)
        rec_size,center2_x,center2_y=helper((i+1)*chip_size-1,(j+1)*chip_size-1,convnet,layer_names,imsize,layer)
        #print ("rec_size ",rec_size)
        #print ("center1_x ",center1_x)
        #print ("center2_x ",center2_x)
        #print ("center1_y ",center1_y)
        #print ("center2_y",center2_y)

        temp2=((-1.0)*torch.ones(5,int(center2_x-center1_x+rec_size),int(center2_y-center1_y+rec_size))).to(device1)
        pad_value=(rec_size)/2
        #for m in range (i*crop_size-pad_value:i*crop_size+crop_size+pad_value):
        #  for n in range (j*crop_size-pad_value:j*crop_size+crop_size+pad_value):
        #    if m>=0 and n>=0 and m<train2_label.shape[0] and n<train2_label.shape[1]:
        #      temp2[:][:][m-(i*crop_size-pad_value)][n-(j*crop_size-pad_value)]=train_data[:][:][m][n]
        #new_train_data.append(temp2)#***************************************
        #print ("ts_data.shape--------------------------",ts_data.shape)
        for m in range (int(center1_x-pad_value),int(center2_x+pad_value)):
          for n in range (int(center1_y-pad_value),int(center2_y+pad_value)):
            #if m>=0 and n>=0 and m<ts_data.shape[1] and n<ts_data.shape[2]:
            if m>=0 and n>=0 and m<2278 and n<2278:
              temp2[:,m-int(center1_x-pad_value),n-int(center1_y-pad_value)]=torch.squeeze(ts_data[:,m,n])
        new_ts_data.append(temp2)
        point_list.append([center1_x-pad_value,center1_y-pad_value])
  print ("len(new_ts_data) ",len(new_ts_data))
  if len(new_ts_data)>0:
    print ("new_ts_data[0].shape",new_ts_data[0].shape)
    print ("new_ts_data[0].size()",new_ts_data[0].size())
  #print ("new_ts_label.shape",(np.array(new_ts_label)).shape)    
  return new_ts_data,point_list#new_ts_label,

def make_final_prediction(predicted_2,point_list):
  #a=torch.zeros(2278,2278)
  final_size=2278
  a=torch.zeros(final_size,final_size).type(torch.FloatTensor)
  count=torch.zeros(final_size,final_size)
  for i in range (0,len(predicted_2)):
    temp=transform.resize(predicted_2[i],(261, 261))
    #print ("temp=",temp)
    start_x=point_list[i][0]
    start_y=point_list[i][1]
    for m in range (int(start_x),int(start_x+261)):
      for n in range (int(start_y),int(start_y+261)):
            #if m>=0 and n>=0 and m<ts_data.shape[1] and n<ts_data.shape[2]:
        if m>=0 and n>=0 and m<final_size and n<final_size:
          a[m,n]+=temp[m-int(start_x),n-int(start_y)]#temp[0:130,0:130]
          count[m,n]+=1
  for i in range(0,final_size):
    for j in range(0,final_size):
      if count[i,j]>1:
        a[i,j]=a[i,j]/count[i,j]


  return a



def transpose_to_pytorch_input(x):
  #print ("x.shape",x.shape)
  y=np.transpose(x,(0,3,1,2))
  return y
def normalize_five_frame_data(data,d_max,d_min):
    #dd=np.expand_dims(d,axis=2)
    #k=np.concatenate((aa,bb,cc,dd),axis=2)

    #print ("data.shape",data.shape)
    num_channels=5
    #print ("data.shape-------------",data.shape)
    data_normalized=np.zeros((num_channels,data.shape[1],data.shape[2]),np.float64)#np.uint8)#,np.float64)
    #print ("data.shape",data.shape)
    for i in range (0, num_channels):
      #print ("hello ",i)
      data_normalized[i,:,:]=(2*(data[i,:,:].astype(np.float64)-d_min)/(d_max-d_min))-1
      #data_normalized[i,:,:]=((preprocessing.minmax_scale(data[i,:,:].squeeze(),feature_range=(-1,1))).astype(np.float64))
    #print ("data_normalized.shape",data_normalized.shape)
    return data_normalized
def normalize_five_frame_label(data,d_max,d_min):
    #dd=np.expand_dims(d,axis=2)
    #k=np.concatenate((aa,bb,cc,dd),axis=2)
    #print ("data.shape",data.shape)
    data_normalized=np.zeros((data.shape[0],data.shape[1]),np.float64)#np.uint8)#,np.float64)
    data_normalized=(data.astype(np.float64)-d_min)/(d_max-d_min)
    #data_normalized=(preprocessing.minmax_scale(data,feature_range=(0,1))).astype(np.float64)#).unsqueeze(0)
    #print ("data_normalized.shape",data_normalized.shape)
    return data_normalized


def count(predicted_ave,target_ave):
  false_positive=0.0
  true_positive=0.0
  false_negative=0.0
  true_negative=0.0
  s_p=predicted_ave.shape
  s_t=target_ave.shape
  print ("s_p[0]",s_p[0])
  num1=s_p[0]
  num2=s_t[0]
  for i in range(0,num1):
    flag_pos=False
    for j in range(0,num2):
      if math.sqrt((predicted_ave[i,0]-target_ave[j,0])**2+(predicted_ave[i,1]-target_ave[j,1])**2)<20:#torch.dist(predicted_ave[i],target_ave[j],p=2)<20:#
        true_positive+=1
        flag_pos=True
        break
    if flag_pos==False:
      false_positive+=1
  for i in range (0,num2):#p1,p2=p.shape
    flag_neg=False
    for j in range (0,num1):
      if math.sqrt((target_ave[i,0]-predicted_ave[j,0])**2+(target_ave[i,1]-predicted_ave[j,1])**2)<20:#torch.dist(target_ave[i],predicted_ave[j],p=2)<20:#
        flag_neg=True
        break
    if flag_neg==False:
      false_negative+=1
  if (true_positive+false_positive)>0:
    precision=true_positive/(true_positive+false_positive)
  else:
    precision=0.0

  if (true_positive+false_negative)>0:
    recall=true_positive/(true_positive+false_negative)
  else:
    recall=0.0
    
  return precision,recall


def prepare_to_compare(predicted,target):

  for i in range(predicted.shape[0]):
    for j in range(predicted.shape[1]):
      if predicted[i,j]>0.2:
        predicted[i,j]=1
      else:
        predicted[i,j]=0
  structure = np.ones((3, 3), dtype=np.float64)
  print ("predicted.shape",predicted.shape)
  print ("target.shape",target.shape)
  print ("type(predicted)",type(predicted))
  print ("type(target)",type(target))

  labeled, ncomponents = label(predicted, structure)
  #print ("labeled[500:640,900:1040]",labeled[500:640,900:1040])
  #print ("ncomponents",ncomponents)
  indices = np.indices(predicted.shape).T[:,:,[1, 0]]
  #print("indices[labeled == 1]",indices[labeled == 97])
  #print (np.mean(indices[labeled == 97],axis=0))
  predicted_averages1=np.zeros((ncomponents,2),dtype = np.float64)
  for i in range(ncomponents):
    predicted_averages1[i]=np.mean(indices[labeled == i+1],axis=0)


  labeled2, ncomponents2 = label(target, structure)
  indices2 = np.indices(target.shape).T[:,:,[1, 0]]
  target_averages2=np.zeros((ncomponents2,2),dtype = np.float64)
  for i in range(ncomponents2):
    target_averages2[i]=np.mean(indices2[labeled2 == i+1],axis=0)
  
  print ("predicted_averages1",predicted_averages1)
  print ("target_averages2",target_averages2)
  print ("predicted_averages1.shape",predicted_averages1.shape)
  print ("target_averages2.shape",target_averages2.shape)

  return predicted_averages1,target_averages2



device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#train_data_path="/home/babak/Desktop/WPAFB2009/AOI01/finalCrop/"

#train_data_path="/home/babak/Desktop/WPAFB2009/AOI01/finalCrop/"
#train_1_label_path="/home/babak/Desktop/WPAFB2009/AOI01/Heatmaps/BinSegClusterHeatmapsRedacted/"
#train_2_label_path="/home/babak/Desktop/WPAFB2009/AOI01/Heatmaps/BinSegHeatmapsRedacted/"



train_data_path="/home/babak/WPAFB2009/benchmark/AOI34_41_42/finalCrop/"
train_1_label_path="/home/babak/WPAFB2009/benchmark/AOI34_41_42/Heatmaps/BinSegClusterHeatmapsRedacted/"
train_2_label_path="/home/babak/WPAFB2009/benchmark/AOI34_41_42/Heatmaps/BinSegHeatmapsRedacted/"


test_data_path="/home/babak/WPAFB2009/benchmark/AOI01/finalCrop/"
test_1_label_path="/home/babak/WPAFB2009/benchmark/AOI01/Heatmaps/BinSegClusterHeatmapsRedacted/"
test_2_label_path="/home/babak/WPAFB2009/benchmark/AOI01/Heatmaps/BinSegHeatmapsRedacted/"



#train_data_path="/home/babak/WPAFB2009/AOI01/finalCrop/"
#train_1_label_path="/home/babak/WPAFB2009/AOI01/Heatmaps/BinSegClusterHeatmapsRedacted/"
#train_2_label_path="/home/babak/WPAFB2009/AOI01/Heatmaps/BinSegHeatmapsRedacted/"


data_temp=read_images(train_data_path,"pgm",False)#1-read_images(train_data_path,"pgm",False)
data = data_temp#add_frames_to_channels(data_temp)
#((data_temp.astype(np.float64)-np.amin(data_temp))/(np.amax(data_temp)-np.amin(data_temp)))
 ###################################data = data.astype('float16')

label1=read_images(train_1_label_path,"pbm",False)#1-read_images(train_label_path,"pbm",False)
label2=read_images_2(train_2_label_path,"pbm",False)#

d_max=255#np.amax(label)
d_min=0#np.amin(label) 


test_data_temp=read_images(test_data_path,"pgm",True)#1-read_images(train_data_path,"pgm",False)
#data = data_temp#add_frames_to_channels(data_temp)
#((data_temp.astype(np.float64)-np.amin(data_temp))/(np.amax(data_temp)-np.amin(data_temp)))
 ###################################data = data.astype('float16')

test_label1=read_images(test_1_label_path,"pbm",True)#1-read_images(train_label_path,"pbm",False)
test_label2=read_images_2(test_2_label_path,"pbm",True)



print ("data.shape",data.shape)
print ("label.shape",label1.shape)
print ("label2.shape",label2.shape)

print ("test_data_temp.shape",test_data_temp.shape)
print ("test_label1.shape",test_label1.shape)
print ("test_label2.shape",test_label2.shape)

#print ("zero values: " ,np.where(train_label[100] == 0))
#a=data[1,:,:,2]*255
#b=label[1]*255
#c=label2[1]*255


#data=transpose_to_pytorch_input(data)



#train_data=data[0:800,:,:,:]
#train_label=label1[0:800,:,:]
#valid_data=data[800:900,:,:,:]
#valid_label=label1[800:900,:,:]
#test_data=data[900:1026,:,:,:]
#test_label=label1[900:1026,:,:]


###train_data=data[0:800,:,:,:]
#train2_label=label2[0:800,:,:]
###valid_data=data[801:900,:,:,:]
#valid2_label=label2[800:900,:,:]
###test_data=data[901:1025,:,:,:]
#test2_label=label2[900:1026,:,:]



train_data=data[0:2049,:,:]
train_label=label1[0:2049,:,:]
train2_label=label2[0:2049,:,:]

#########train_data=data[0:100,:,:]
#########train_label=label1[0:100,:,:]
########train2_label=label2[0:100,:,:]

test_data=test_data_temp[0:1025,:,:]
test_label=test_label1[0:1025,:,:]
test2_label=test_label2[0:1025,:,:]

#test_data=test_data_temp[0:100,:,:]
#test_label=test_label1[0:100,:,:]
#test2_label=test_label2[0:100,:,:]





#print ("------------------------max_train2_label",np.amax(train2_label))

#print ("train_data[0,0,10:30,10:30]",train_data[0,0,10:30,10:30])



#print ("train_data.shape",train_data.shape)
#print ("train_label.shape",train_label.shape)
#print ("train2_label.shape",train2_label.shape)



print ("test_data.shape",test_data.shape)
print ("test_label.shape",test_label.shape)
print ("test2_label.shape",test2_label.shape)
#print ("valid_data.shape",valid_data.shape)
#print ("valid_label.shape",valid_label.shape)
#print ("test_data.shape",test_data.shape)
#print ("test_label.shape",test_label.shape)

#shuffled_list=np.arange(800)
shuffled_list=np.arange(train_data.shape[0])#100)#2049)

np.random.shuffle(shuffled_list)
#print ("shuffled_list",shuffled_list)



num_epochs = 12
#num_classes = 10
batch_size = 8
#learning_rate = 0.001
learning_rate = 0.001


'''

model = ConvNet().to(device0)
#model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.MSELoss()#nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


train_loss_epoch=[]
train_loss_batch=[]
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
total_batches=train_data.shape[0]//batch_size
im_size_input=2278
#batch_train_data_1=np.zeros((batch_size,train_data.shape[1],train_data.shape[2],train_data.shape[3]),np.float64)
batch_train_data_1=np.zeros((batch_size,5,train_data.shape[1],train_data.shape[2]),np.float64)
batch_train_label_1=np.zeros((batch_size,train_label.shape[1],train_label.shape[2]),np.float64)

model.train()
for i in range(num_epochs):
  for batch_idx in range(0,total_batches):
    k=0
    while (k<batch_size):
      print ("batch_idx= ",batch_idx)
      #print ("shuffled_list[batch_idx*batch_size+k]",shuffled_list[batch_idx*batch_size+k])
      #print ("train_data[shuffled_list[batch_idx*batch_size+k],:,:,:].shape",train_data[shuffled_list[batch_idx*batch_size+k],:,:,:].shape)
      #print ("normalize_five_frame_data(train_data[shuffled_list[batch_idx*batch_size+k],:,:,:]",normalize_five_frame_data(train_data[shuffled_list[batch_idx*batch_size+k],:,:,:]))
      #print ("shuffled_list[batch_idx*batch_size+k]",shuffled_list[batch_idx*batch_size+k])
      iindex=shuffled_list[batch_idx*batch_size+k]
      #batch_train_data_1[k,:,:,:] = np.expand_dims(normalize_five_frame_data(add_frames_to_channels2(train_data[iindex,:,:],iindex),d_max,d_min),axis=0)
      batch_train_data_1[k,:,:,:] = np.expand_dims(normalize_five_frame_data(add_frames_to_channels2(train_data,iindex),d_max,d_min),axis=0)
      
      #batch_train_data_1[k,:,:,:] = normalize_five_frame_data(train_data[shuffled_list[batch_idx*batch_size+k],:,:,:])
      #print ("**********************************************************************************************")
      #print ("batch_train_data.shape",batch_train_data_1.shape)
      batch_train_label_1[k,:,:] = np.expand_dims(normalize_five_frame_label(train_label[iindex,:,:],d_max,d_min),axis=0) 
      
      
      #print ("batch_train_label.shape",batch_train_label_1.shape)
      k+=1
    #sys.exit("Error message")
    print ("batch_train_data_1.shape",batch_train_data_1.shape)
    print ("batch_train_label_1.shape",batch_train_label_1.shape)
    
    batch_train_data_1_tensor=(torch.from_numpy(batch_train_data_1)).type(torch.FloatTensor)
    batch_train_label_1_tensor =(torch.from_numpy(batch_train_label_1)).type(torch.FloatTensor)#.type(torch.DoubleTensor)#.astype(np.float32))  
    images=batch_train_data_1_tensor.to(device0)
    labels=batch_train_label_1_tensor.to(device0)#.cuda()
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    
        
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Iter= " + str(i) + " batch_idx= "+str( batch_idx)+", Loss= " + "{:.6f}".format(loss.item()) + ", Training Accuracy= " )#+ "{:.5f}".format(acc))

    train_loss.append(loss.item())
    #train_loss_epoch.append()
    train_loss_batch.append(i*total_batches+batch_idx)

    if ((batch_idx+1)%200==0):
      torch.save(model.state_dict(),'/home/babak/WPAFB2009/benchmark/AOI34_41_42/network1/model1_'+str(i)+'_'+str(batch_idx)+'.ckpt') 


    #test_loss.append(valid_loss)
    #train_accuracy.append(acc)
    #test_accuracy.append(test_acc)
print("Optimization Finished!")


train_loss_np=np.asarray(train_loss, dtype=np.float64)
    #train_loss_epoch.append()
train_loss_batch_np=np.asarray(train_loss_batch, dtype=np.float64)

plt.plot(train_loss_batch_np, train_loss_np)
plt.xlabel('batch_idx (batchsize=8)')
plt.ylabel('loss')
plt.title('Clusternet train loss function ')
plt.savefig("/home/babak/WPAFB2009/benchmark/AOI34_41_42/network1/train_clusternet_error12.png", bbox_inches='tight')


all_precision_1=[]
all_recall_1=[]


#sys.exit("Error message")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i in range (0,test_data.shape[0]):
        print("test_Data ",i)
        #images_test = torch.from_numpy(test_data[i].astype(np.float32)).unsqueeze_(0).to(device0)
        images_test = (torch.from_numpy(normalize_five_frame_data(add_frames_to_channels2(test_data,i),d_max,d_min).astype(np.float64))).unsqueeze_(0).type(torch.FloatTensor).to(device0)
        #print ("images_test.shape",images_test.shape)
        labels_test = (torch.from_numpy(normalize_five_frame_label(test_label[i],d_max,d_min).astype(np.float64))).unsqueeze_(0).type(torch.FloatTensor).to(device0)#.unsqueeze_(0).to(device0)#.float32)).to(device)
        #print ("labels_test.shape",labels_test.shape)
        #print(images_test.size())
        with torch.no_grad():outputs = model(images_test)
        predicted = outputs.data#torch.max(outputs.data, 1)
        #if(i==test_data.shape[0]-1):
       	if(i==0):
          #########################print ("test_label[0]",test_label[0])
          f=predicted#[0]#.cpu().numpy()
          print ("f.size()",f.size())
          cv2.imwrite('fist_test_image12.png',f.cpu().numpy()*255)
          #cv2.destroyAllWindows()
          #print(predicted[0])
          predicted_averages,target_averages=prepare_to_compare(predicted.cpu().numpy(),labels_test.squeeze_().cpu().numpy())
          precision,recall=count(predicted_averages,target_averages)
          print("precision",precision)
          print ("recall",recall)
          all_precision_1.append(precision)
          all_recall_1.append(recall)
          #sys.exit("Error message")
        #total += (labels_test.size(0)*labels_test.size(1))
        #correct += (predicted == labels_test).sum().item()

    #print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')



#*****************************************************************************************************************


num_epochs2 = 14
#num_classes = 10
batch_size2 = 32
#learning_rate = 0.001
learning_rate2 = 0.0001


model2 = FoveaNet().to(device1)
#model = nn.DataParallel(model)

# Loss and optimizer
criterion2 = nn.MSELoss()#nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate2,weight_decay=1e-5)

#loss = (y_pred - y).pow(2).sum()

train_loss2_epoch=[]
train_loss2_batch=[]

train_loss2 = []
test_loss2 = []
train2_accuracy = []
test2_accuracy = []
train2_data=[]
test2_data=[]
train2_label_cropped=[]
total2_batches=train_data.shape[0]//batch_size2

model2.train()
for i in range(5):
  j=0;
  while j < train_label.shape[0]:
    #train_cropped_data,train_cropped_label,position=make_train_data_fovea(train_data[j],train2_label[j],j)
    #train2_data.extend(train_cropped_data)
    #train2_label_cropped.extend(train_cropped_label)
    #j+=1
    
    while (len(train2_data)< batch_size2) and (j < train_label.shape[0]):
      
      #train_cropped_data,train_cropped_label,position=make_train_data_fovea(train_data[j],train2_label[j],j)#.astype(np.float64)
      #train_cropped_data,train_cropped_label,position=make_train_data_fovea(normalize_five_frame_data(train_data[shuffled_list[j],:,:,:],255,0),normalize_five_frame_label(train2_label[shuffled_list[j],:,:],255,0),shuffled_list[j])
      #train2_lab=cv2.resize(normalize_five_frame_label(train2_label[shuffled_list[j],:,:],255,0), dsize=(2340, 2340))
      train_cropped_data,train_cropped_label,position=make_train_data_fovea(normalize_five_frame_data(add_frames_to_channels2(train_data,shuffled_list[j]),255,0),normalize_five_frame_label(train_label[shuffled_list[j],:,:],255,0),normalize_five_frame_label(train2_label[shuffled_list[j],:,:],255,0),shuffled_list[j])
      if i==0 and j==4:
        for al in range (0, len(train_cropped_data)):
          cv2.imwrite('/home/babak/WPAFB2009/benchmark/AOI34_41_42/tr/train_data_'+str(i)+'_'+str(j)+'_'+str(al)+'.png',((train_cropped_data[al][2]+1.0)/2)*255)
          cv2.imwrite('/home/babak/WPAFB2009/benchmark/AOI34_41_42/tr/train_label_'+str(i)+'_'+str(j)+'_'+str(al)+'.png',train_cropped_label[al]*255)

      print ("len(train_cropped_data)",len(train_cropped_data))
      print ("len(train_cropped_label)",len(train_cropped_label))
      print ("train_cropped_data.shape",train_cropped_data[0].shape)
      print ("train_cropped_label.shape",train_cropped_label[0].shape)
      #cv2.imwrite('/home/babak/WPAFB2009/GI/train_cropped_data'+str(j)+'.png',((train_cropped_data[0][2]+1.0)/2)*255)#.cpu().numpy()*255)
      #cv2.imwrite('/home/babak/WPAFB2009/GI/train_cropped_label'+str(j)+'.png',train_cropped_label[0]*255)#.cpu().numpy()*255)
      #print ("train_cropped_data[0]",(train_cropped_data[0])[2,0:20,0:20])
      #print ("train_cropped_label[0]",(train_cropped_label[0])[0:20,0:20])
      #print ("position",position)
      #sys.exit("Error message")

      train2_data.extend(train_cropped_data)
      train2_label_cropped.extend(train_cropped_label)
      j+=1
      
         #for batch_idx in range(0,total_batches):
    print ("train2_data.shape",(np.array(train2_data)).shape)
    print ("train2_label_cropped.shape",(np.array(train2_label_cropped)).shape)
    batch_train2_data = np.array(train2_data)[0:batch_size2]#,:,:,:]#train_data.shape[0])]
    print ("batch_train2_data.shape",batch_train2_data.shape)
    batch_train2_label = np.array(train2_label_cropped)[0:batch_size2]#,:,:,:]#:min((batch_idx+1)*batch_size,:,:,:]#,train_label.shape[0])]    
    print ("batch_train2_label.shape",batch_train2_label.shape)
    #print("j=",j)
    #print ("batch_train2_data.shape[0]",batch_train2_data.shape[0])
    #if j<2: 
    #  for bag in range (0,batch_train2_data.shape[0]):
    #    print ("bag",bag)
    #    cv2.imwrite('/home/babak/WPAFB2009/GI/train_cropped_data'+str(j)+'_'+str(bag)+'.png',((batch_train2_data[bag][2]+1.0)/2)*255)#.cpu().numpy()*255)
    #    cv2.imwrite('/home/babak/WPAFB2009/GI/train_cropped_label'+str(j)+'_'+str(bag)+'.png',batch_train2_label[bag]*255)#.cpu().numpy()*255)
    #train2_data=np.delete(np.array(train2_data), np.s_[0:batch_size2], axis=0)
    #train2_label_cropped=np.delete(np.array(train2_label_cropped), np.s_[0:batch_size2], axis=0)
    #np.delete(np.array(train2_data), np.s_[0:batch_size2], axis=0)
    #np.delete(np.array(train2_label_cropped), np.s_[0:batch_size2], axis=0)
    #sys.exit("Error message")
    
    batch_train2_data_tensor=torch.from_numpy(batch_train2_data.astype(np.float64)).type(torch.FloatTensor)
    batch_train2_label_tensor =torch.from_numpy(batch_train2_label.astype(np.float64)).type(torch.FloatTensor)#.astype(np.float64))  
    images2=batch_train2_data_tensor.to(device1)
    labels2=batch_train2_label_tensor.to(device1)#.cuda()
    #print(images2.size(),"ssssssssssssssssssssss")

    outputs2 = model2(images2)
    loss2 = criterion2(outputs2, labels2)
        
    # Backward and optimize
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()

    print("Iter= " + str(i) + " batch_idx= "+str(j)+", Loss= " + "{:.6f}".format(loss2.item()) + ", Training Accuracy= " )#+ "{:.5f}".format(acc))

    del train2_data[0:batch_size2]
    del train2_label_cropped[0:batch_size2]
    #print ("train2_data.shape",(np.array(train2_data)).shape)
    #print ("train2_label_cropped.shape",(np.array(train2_label_cropped)).shape)

    train_loss2.append(loss2.item())
    #train_loss2_epoch.append(i*train_label.shape[0]+j)
    train_loss2_batch.append(i*train_label.shape[0]+j)
    train_loss2_batch_np=np.asarray(train_loss2_batch, dtype=np.float64)

    #if j==100:
    #  break
    
    if ((j+1)%200==0):
      torch.save(model2.state_dict(), '/home/babak/WPAFB2009/benchmark/AOI34_41_42/network2/model2_'+str(i)+'_'+str(j)+'.ckpt') 


      #test_loss.append(valid_loss)
      #train_accuracy.append(acc)
      #test_accuracy.append(test_acc)
  #torch.save(model2.state_dict(), 'model2_'+str(i)+'.ckpt') 
  print("Optimization Finished!")
train_loss2_np=np.asarray(train_loss2, dtype=np.float32)
train_loss2_batch_np=np.asarray(train_loss2_batch, dtype=np.float64)


#sys.exit("Error message")

#true_positive=0
#false_positive=0
#false_negative=0




'''


model = ConvNet()
model.load_state_dict(torch.load('/home/babak/WPAFB2009/benchmark/AOI34_41_42/network1/model1_9_199.ckpt'))#.cuda()
model.to(device1)



model2 = FoveaNet()
model2.load_state_dict(torch.load('/home/babak/WPAFB2009/benchmark/AOI34_41_42/network2/model2_9_1799.ckpt'))
model2.to(device1)


model.eval()
model2.eval()

all_precision_2=[]
all_recall_2=[]


final_prediction=[]

with torch.no_grad():
    correct = 0
    total = 0
    for i in range (0,test_data.shape[0]):
        predicted2=[]
        images_test = torch.from_numpy(normalize_five_frame_data(add_frames_to_channels2(test_data,i).astype(np.float64),d_max,d_min)).unsqueeze_(0).type(torch.FloatTensor).to(device1)

        labels_test = torch.from_numpy(normalize_five_frame_label(test_label[i].astype(np.float64),d_max,d_min)).unsqueeze_(0).type(torch.FloatTensor).to(device1)#.float32)).to(device)
        labels2_test = torch.from_numpy(normalize_five_frame_label(test2_label[i].astype(np.float64),d_max,d_min)).unsqueeze_(0).type(torch.FloatTensor).to(device1)
        outputs = model(images_test)

        predicted = outputs.data#torch.max(outputs.data, 1)
        cv2.imwrite('/home/babak/WPAFB2009/benchmark/AOI34_41_42/GI2/output_first_net'+str(i)+'.png',predicted.round_().cpu().numpy()*255)
        #print ("images_test.size()",images_test.size())
        #print ("predicted.size()",predicted.size())
      
        #images_test.size() (1, 5, 2278, 2278)
        #predicted.size() (72, 72)
        #np.expand_dims(,axis=0)
        new_test_data,point_list=make_test_data_fovea(images_test.squeeze_(),labels_test.squeeze_().cpu().numpy(),predicted.round_().cpu().numpy(),i)
        #new_test_data,new_test_label,point_list=make_test_data_fovea(images_test.squeeze_(),labels_test.squeeze_().cpu().numpy(),predicted.round_().cpu().numpy(),i)
        #new_test_data,new_test_label,point_list=make_test_data_fovea(normalize_five_frame_data(images_test.squeeze_().cpu().numpy(),d_max,d_min),normalize_five_frame_label(labels_test.squeeze_().cpu().numpy(),d_max,d_min),normalize_five_frame_label(predicted.cpu().numpy(),d_max,d_min),i)
        
        
        #if i<1: 
            
        #    for bag in range (0,len(new_test_data)):
        #      print ("new_test_data[i].size()",new_test_data[i].size())
        #      print ("bag",bag)

        #      cv2.imwrite('/home/babak/WPAFB2009/GI/test_cropped_data'+str(i)+'_'+str(bag)+'.png',(((new_test_data[bag][2]+1.0)/2)*255).cpu().numpy())#.cpu().numpy()*255)
        #      print ("point_list"+str(bag)+":",point_list[bag])
        #      #cv2.imwrite('/home/babak/WPAFB2009/GI/test_cropped_label'+str(j)+'_'+str(bag)+'.png',batch_train2_label[bag]*255)#.cpu().numpy()*255)

        
        print("len(new_test_data)",len(new_test_data))
        #new_test_data2=torch.from_numpy((np.array(new_test_data)).astype(np.float32)).unsqueeze_(0).to(device1)
        for j in range (0,len(new_test_data)):#new_test_data.shape[0]):
          #print ('new_test_data[j].shape',new_test_data[j].shape)
          #print ('new_test_data[j].size',new_test_data[j].size)
          outputs2=model2(new_test_data[j].unsqueeze_(0))
          #print ('outputs2.shape',outputs2.shape)
          predicted2.append(outputs2.data)

          if i==0:
            print ("i,j=",i,j)
            print ("point_list "+str(j)+" : ",point_list[j][0],point_list[j][1])
            cv2.imwrite('/home/babak/WPAFB2009/benchmark/AOI34_41_42/GI2/outputs_second_net'+str(i)+'_'+str(j)+'.png',outputs2.data.cpu().numpy()*255)
        #sys.exit("Error message")
        final_prediction=make_final_prediction(predicted2,point_list)
        print ("---------------------------- i = ",i)
        #if (i==0):
        cpu_final_prediction=final_prediction.cpu().numpy()#.round_().cpu().numpy()
        resized_cpu_final_prediction=cv2.resize(cpu_final_prediction, dsize=(2278, 2278))
        keep_image=final_prediction.round_()
        print ("keep_image.size()",keep_image.size())
        #cv2.imwrite('image_final_alaki.png',keep_image.cpu().numpy()*255)
        #predicted_averages,target_averages=prepare_to_compare(final_prediction.cpu().numpy(),labels2_test.squeeze_().cpu().numpy())
        cv2.imwrite('/home/babak/WPAFB2009/benchmark/AOI34_41_42/GI2/resized_cpu_final_prediction'+str(i)+'.png',resized_cpu_final_prediction*255)#i+900)+'.png',resized_cpu_final_prediction*255)#.cpu().numpy()*255)
        predicted_averages,target_averages=prepare_to_compare(resized_cpu_final_prediction,labels2_test.squeeze_().cpu().numpy())
        precision,recall=count(predicted_averages,target_averages)
        print("precision",precision)
        print ("recall",recall)

        print("final_prediction.shape=",final_prediction.shape)
        print("labels2_test.shape=",labels2_test.shape)

        all_precision_2.append(precision)
        all_recall_2.append(recall)
        #t_p,f_p,f_n=count(torch.round(final_prediction),labels2_test)

print ("average precision= ", sum(all_precision_2)/len(all_precision_2))
print ("average recall= ", sum(all_recall_2)/len(all_recall_2))

# Save the model checkpoint

#cv2.imwrite("predicted.png", keep)
#torch.save(model.state_dict(), 'model2.ckpt')


