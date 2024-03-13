#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('git clone https://github.com/natanielruiz/deep-head-pose.git dhp')


# In[4]:


get_ipython().system('git clone https://github.com/natanielruiz/deep-head-pose.git dhp')


# In[5]:


get_ipython().run_line_magic('cd', 'dhp/code')


# In[6]:


get_ipython().system('pip install mtcnn')


# In[7]:


import numpy as np
import torch
#from torch.utils.serialization import load_lua
import os
import scipy.io as sio
import cv2
import math
from math import cos, sin

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def mse_loss(input, target):
    return torch.sum(torch.abs(input.data - target.data) ** 2)

def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


# In[8]:


import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F


#import datasets, hopenet, utils
import hopenet
import utils
from mtcnn import MTCNN
import cv2

from PIL import Image
#from google.colab.patches import cv2_imshow
#import cv2_imshow
import numpy as np



cudnn.enabled = True
gpu = 0
snapshot_path = "/home/snake/Desktop/zhansaya/emotion classification/dhp/hopenet_robust_alpha1.pkl"


# In[10]:


# ResNet50 structure
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)


# Load snapshot
saved_state_dict = torch.load(snapshot_path)
model.load_state_dict(saved_state_dict)



transformations = transforms.Compose([transforms.Resize(224),
transforms.CenterCrop(224), transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


model.cuda(gpu)



# Test the Model
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
total = 0

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

yaw_error = .0
pitch_error = .0
roll_error = .0

l1loss = torch.nn.L1Loss(size_average=False)

pitchlist = []
yawlist = []
rolllist=[]

def calculate_pyr(path):
    img = Image.open(path)
    
    img1 = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    #assuming one face
    face=detector.detect_faces(img1)[0]['box']
    x,y,w,h=face
    #print(x,y,w,h)
    
    img=img.crop((int(x-20),int(y-20),int(x+w+20),int(y+h+20)))

       
    
    img = img.convert('RGB')
    
    cv2_img=np.asarray(img)
    #
    #print(cv2_img.shape)
    cv2_img=cv2.resize(cv2_img,(224,224))[:,:,::-1]
    cv2_img = cv2_img.astype(np.uint8).copy() 
    img = transformations(img)
    
    img=img.unsqueeze(0)
    
    images = Variable(img).cuda(gpu)
    

    yaw, pitch, roll = model(images)

    # Binned predictions
    _, yaw_bpred = torch.max(yaw.data, 1)
    _, pitch_bpred = torch.max(pitch.data, 1)
    _, roll_bpred = torch.max(roll.data, 1)

    # Continuous predictions
    yaw_predicted = utils.softmax_temperature(yaw.data, 1)
    pitch_predicted = utils.softmax_temperature(pitch.data, 1)
    roll_predicted = utils.softmax_temperature(roll.data, 1)

    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
    roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

    
  
    pitch = pitch_predicted[0]
    yaw = -yaw_predicted[0] 
    roll = roll_predicted[0] 
    
    return pitch,yaw,roll
    
    #print("pitch,yaw,roll",pitch,yaw,roll)      
    #utils.draw_axis(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], size=100)
    #cv2.imwrite('/home/snake/Desktop/zhansaya/emotion classification/test/res/311412res.jpg', cv2_img)
    
    #pitchlist.append(pitch)
    #yawlist.append(yaw)
    #rolllist.append(roll)

#put the path of your image here, result will be saved as /content/res.jpg

calculate_pyr('/home/snake/Desktop/5142466/Personne01/personne01122-30+30.jpg')

#test("/home/snake/Desktop/zhansaya/emotion classification/test/right.jpeg")


# In[15]:


#Put all images in one folder
anger_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Anger/"
anger_images = [f for f in os.listdir(anger_folder_path) if os.path.isfile(os.path.join(anger_folder_path, f))]

#all images with the paths
anger_img_paths=[]
for i in range(len(anger_images)):
    anger_img_paths.append(anger_folder_path+anger_images[i])


# In[9]:


#turn all images into grey
from PIL import Image
for i in range(len(anger_img_paths)):
    img = Image.open(anger_img_paths[i]).convert('L')
    img.save(anger_img_paths[i])


# In[21]:


#anger_img_pyr is list with true samples where pyr could be calculated
#bad_anger is list of bad samples, where pyr cannot be calculated
anger_img_pyr=[]
bad_anger=[]

for i in range(len(anger_img_paths)):
    try:
        anger_img_pyr.append(calculate_pyr(anger_img_paths[i]))
    except IndexError:
        print("Index Error: "+ anger_img_paths[i])
        bad_anger.append(anger_img_paths[i])


# In[10]:


anger_dict={}
for key in anger_img_paths:
    for value in anger_img_pyr:
        anger_dict[key] = value
        anger_img_pyr.remove(value)
        break


# In[13]:


with open("/home/snake/Desktop/zhansaya/emotion_dictionaries/anger.txt", 'w') as f: 
    for key, value in anger_dict.items(): 
        f.write('%s:%s\n' % (key, value))


# In[ ]:


len(anger_img_pyr)


# In[ ]:


len(bad_anger)


# In[ ]:


anger_img_pyr[0]


# In[ ]:


calculate_pyr(anger_img_paths[0])


# In[31]:


#Put all images in one folder
contempt_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Contempt/"
contempt_images = [f for f in os.listdir(contempt_folder_path) if os.path.isfile(os.path.join(contempt_folder_path, f))]

#all images with the paths
contempt_img_paths=[]
for i in range(len(contempt_images)):
    contempt_img_paths.append(contempt_folder_path+contempt_images[i])


# In[32]:


contempt_img_pyr=[]
bad_contempt=[]

for i in range(len(contempt_img_paths)):
    try:
        contempt_img_pyr.append(calculate_pyr(contempt_img_paths[i]))
    except IndexError:
        print("Index Error: "+ contempt_img_paths[i])
        bad_contempt.append(contempt_img_paths[i])


# In[33]:


contempt_dict={}
for key in contempt_img_paths:
    for value in contempt_img_pyr:
        contempt_dict[key] = value
        contempt_img_pyr.remove(value)
        break


# In[34]:


with open("/home/snake/Desktop/zhansaya/emotion_dictionaries/contempt.txt", 'w') as f: 
    for key, value in contempt_dict.items(): 
        f.write('%s:%s\n' % (key, value))


# In[35]:


#Put all images in one folder
disgust_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Disgust/"
disgust_images = [f for f in os.listdir(disgust_folder_path) if os.path.isfile(os.path.join(disgust_folder_path, f))]

#all images with the paths
disgust_img_paths=[]
for i in range(len(disgust_images)):
    disgust_img_paths.append(disgust_folder_path+disgust_images[i])


# In[36]:


disgust_img_pyr=[]
bad_disgust=[]

for i in range(len(disgust_img_paths)):
    try:
        disgust_img_pyr.append(calculate_pyr(disgust_img_paths[i]))
    except IndexError:
        print("Index Error: "+ disgust_img_paths[i])
        bad_disgust.append(disgust_img_paths[i])


# In[37]:


disgust_dict={}
for key in disgust_img_paths:
    for value in disgust_img_pyr:
        disgust_dict[key] = value
        disgust_img_pyr.remove(value)
        break


# In[38]:


with open("/home/snake/Desktop/zhansaya/emotion_dictionaries/disgust.txt", 'w') as f: 
    for key, value in disgust_dict.items(): 
        f.write('%s:%s\n' % (key, value))


# In[39]:


#Put all images in one folder
fear_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Fear/"
fear_images = [f for f in os.listdir(fear_folder_path) if os.path.isfile(os.path.join(fear_folder_path, f))]

#all images with the paths
fear_img_paths=[]
for i in range(len(fear_images)):
    fear_img_paths.append(fear_folder_path+fear_images[i])


# In[40]:


fear_img_pyr=[]
bad_fear=[]

for i in range(len(fear_img_paths)):
    try:
        fear_img_pyr.append(calculate_pyr(fear_img_paths[i]))
    except IndexError:
        print("Index Error: "+ fear_img_paths[i])
        bad_fear.append(fear_img_paths[i])


# In[41]:


fear_dict={}
for key in fear_img_paths:
    for value in fear_img_pyr:
        fear_dict[key] = value
        fear_img_pyr.remove(value)
        break


# In[42]:


with open("/home/snake/Desktop/zhansaya/emotion_dictionaries/fear.txt", 'w') as f: 
    for key, value in fear_dict.items(): 
        f.write('%s:%s\n' % (key, value))


# In[43]:


#Put all images in one folder
happy_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Happy/"
happy_images = [f for f in os.listdir(happy_folder_path) if os.path.isfile(os.path.join(happy_folder_path, f))]

#all images with the paths
happy_img_paths=[]
for i in range(len(happy_images)):
    happy_img_paths.append(happy_folder_path+happy_images[i])


# In[44]:


happy_img_pyr=[]
bad_happy=[]

for i in range(len(happy_img_paths)):
    try:
        happy_img_pyr.append(calculate_pyr(happy_img_paths[i]))
    except IndexError:
        print("Index Error: "+ happy_img_paths[i])
        bad_happy.append(happy_img_paths[i])


# In[45]:


happy_dict={}
for key in happy_img_paths:
    for value in happy_img_pyr:
        happy_dict[key] = value
        happy_img_pyr.remove(value)
        break


# In[46]:


with open("/home/snake/Desktop/zhansaya/emotion_dictionaries/happy.txt", 'w') as f: 
    for key, value in happy_dict.items(): 
        f.write('%s:%s\n' % (key, value))


# In[16]:


#Put all images in one folder
neutral_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Neutral/"
neutral_images = [f for f in os.listdir(neutral_folder_path) if os.path.isfile(os.path.join(neutral_folder_path, f))]

#all images with the paths
neutral_img_paths=[]
for i in range(len(neutral_images)):
    neutral_img_paths.append(neutral_folder_path+neutral_images[i])


# In[17]:


neutral_img_pyr=[]
bad_neutral=[]

for i in range(len(neutral_img_paths)):
    try:
        neutral_img_pyr.append(calculate_pyr(neutral_img_paths[i]))
    except IndexError:
        print("Index Error: "+ neutral_img_paths[i])
        bad_neutral.append(neutral_img_paths[i])


# In[18]:


neutral_dict={}
for key in neutral_img_paths:
    for value in neutral_img_pyr:
        neutral_dict[key] = value
        neutral_img_pyr.remove(value)
        break


# In[19]:


with open("/home/snake/Desktop/zhansaya/emotion_dictionaries/neutral.txt", 'w') as f: 
    for key, value in neutral_dict.items(): 
        f.write('%s:%s\n' % (key, value))


# In[12]:


#Put all images in one folder
sad_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Sad/"
sad_images = [f for f in os.listdir(sad_folder_path) if os.path.isfile(os.path.join(sad_folder_path, f))]

#all images with the paths
sad_img_paths=[]
for i in range(len(sad_images)):
    sad_img_paths.append(sad_folder_path+sad_images[i])


# In[13]:


sad_img_pyr=[]
bad_sad=[]

for i in range(len(sad_img_paths)):
    try:
        sad_img_pyr.append(calculate_pyr(sad_img_paths[i]))
    except IndexError:
        print("Index Error: "+ sad_img_paths[i])
        bad_sad.append(sad_img_paths[i])


# In[14]:


sad_dict={}
for key in sad_img_paths:
    for value in sad_img_pyr:
        sad_dict[key] = value
        sad_img_pyr.remove(value)
        break


# In[15]:


with open("/home/snake/Desktop/zhansaya/emotion_dictionaries/sad.txt", 'w') as f: 
    for key, value in sad_dict.items(): 
        f.write('%s:%s\n' % (key, value))


# In[10]:


#Put all images in one folder
surprise_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Surprise/"
surprise_images = [f for f in os.listdir(surprise_folder_path) if os.path.isfile(os.path.join(surprise_folder_path, f))]

#all images with the paths
surprise_img_paths=[]
for i in range(len(surprise_images)):
    surprise_img_paths.append(surprise_folder_path+surprise_images[i])


# In[11]:


surprise_img_pyr=[]
bad_surprise=[]

for i in range(len(surprise_img_paths)):
    try:
        surprise_img_pyr.append(calculate_pyr(surprise_img_paths[i]))
    except IndexError:
        print("Index Error: "+ surprise_img_paths[i])
        bad_surprise.append(surprise_img_paths[i])


# In[17]:


surprise_dict={}
for key in surprise_img_paths:
    for value in surprise_img_pyr:
        surprise_dict[key] = value
        surprise_img_pyr.remove(value)
        break


# In[18]:


surprise_dict


# In[19]:


with open("/home/snake/Desktop/zhansaya/emotion_dictionaries/surprise.txt", 'w') as f: 
    for key, value in surprise_dict.items(): 
        f.write('%s:%s\n' % (key, value))


# In[ ]:




