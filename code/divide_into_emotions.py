#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
torch.cuda.is_available()


# In[2]:


if torch.cuda.is_available(): 
     dev = "cuda:0" 
else: 
     dev = "cpu" 
device = torch.device(dev) 
a = torch.zeros(4,3) 
a = a.to(device)


# In[3]:


print(device)


# In[4]:


import pandas as pd


# In[5]:


#directory = '/mnt/nas1/Zhansaya_affectnet/train_set/annotations/'


# In[1]:


directory = '/mnt/nas4/Zhansaya_affectnet/train_set/annotations/'


# In[2]:


land_list=[]


# In[7]:


exp_list=[]


# In[8]:


import glob


# In[9]:


import numpy as np


# In[10]:


for filename in glob.iglob(f'{directory}/*'):
    if filename[-7:] == 'exp.npy':
        exp_list.append(filename)


# In[13]:


len(exp_list)


# In[11]:


# open file in write mode
with open(r'/home/snake/Desktop/zhansaya/emotion classification/expnames.txt', 'w') as fp:
    for file in exp_list:
        # write each item on a new line
        fp.write("%s\n" % file)
    print('Done')


# In[14]:


with open(r"/home/snake/Desktop/zhansaya/emotion classification/expnames.txt", 'r') as fp:
    x = len(fp.readlines())
    print('Total lines:', x) # 8


# In[15]:


exp_list


# In[16]:


expdict = {}
for file in exp_list:
    expdict.update({file[51:-8]: np.load(file,allow_pickle=True)})


# In[21]:


with open(r'/home/snake/Desktop/zhansaya/emotion classification/allfiles.txt','w') as f: 
    for key, value in expdict.items(): 
        f.write('%s:%s\n' % (key, value))
    print('Done')


# In[22]:


with open(r"/home/snake/Desktop/zhansaya/emotion classification/allfiles.txt", 'r') as fp:
    x = len(fp.readlines())
    print('Total lines:', x) # 8


# In[23]:


expdict


# In[26]:


import os, os.path, shutil


# In[50]:


folder_path = "/mnt/nas4/Zhansaya_affectnet/train_set/images/"

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


# In[51]:


len(images)


# 0: Neutral, 1: Happy, 2: Sad, 3:
# Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt

# In[52]:


num=len(images)//3
new_path= '/mnt/nas4/Zhansaya_affectnet/train_set/train52k/'
for i in range(num):
    old_image_path = os.path.join(folder_path, images[i])
    new_image_path = os.path.join(new_path, images[i])
    shutil.move(old_image_path, new_image_path)


# In[53]:


folder_path2 = "/mnt/nas4/Zhansaya_affectnet/train_set/train52k/"

images2 = [f for f in os.listdir(folder_path2) if os.path.isfile(os.path.join(folder_path2, f))]


# In[54]:


len(images2)


# In[55]:


for image in images2:
    img_id = image.split('.')[0]
    for image_id, emotion_id in expdict.items():
        if img_id==image_id:
            emotion_id = emotion_id.astype(int)
            if emotion_id == 0:
                f_name='Neutral'
                new_path2 = os.path.join(folder_path2, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(folder_path2, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 1:
                f_name='Happy'
                new_path2 = os.path.join(folder_path2, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(folder_path2, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 2:
                f_name='Sad'
                new_path2 = os.path.join(folder_path2, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(folder_path2, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 3:
                f_name='Surprise'
                new_path2 = os.path.join(folder_path2, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(folder_path2, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 4:
                f_name='Fear'
                new_path2 = os.path.join(folder_path2, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(folder_path2, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 5:
                f_name='Disgust'
                new_path2 = os.path.join(folder_path2, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(folder_path2, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 6:
                f_name='Anger'
                new_path2 = os.path.join(folder_path2, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(folder_path2, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 7:
                f_name='Contempt'
                new_path2 = os.path.join(folder_path2, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(folder_path2, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                


# In[44]:


a=r'/mnt/nas1/Zhansaya_affectnet/train_set/train10k/Neutral/'


# In[47]:


a[48:]


# In[48]:


import os
#0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt

paths=[r'/mnt/nas4/Zhansaya_affectnet/train_set/train39k/Neutral/',
       r'/mnt/nas4/Zhansaya_affectnet/train_set/train39k/Happy/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train39k/Sad/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train39k/Surprise/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train39k/Fear/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train39k/Disgust/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train39k/Anger/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train39k/Contempt/']

for path in paths:
    dir_path= path
    print(path[48:],len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))


# In[49]:


import os
#0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt

paths=[r'/mnt/nas4/Zhansaya_affectnet/train_set/train10651/Neutral/',
       r'/mnt/nas4/Zhansaya_affectnet/train_set/train10651/Happy/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train10651/Sad/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train10651/Surprise/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train10651/Fear/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train10651/Disgust/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train10651/Anger/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train10651/Contempt/']

for path in paths:
    dir_path= path
    print(path[48:],len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))


# In[1]:


import os
#0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt

paths=[r'/mnt/nas4/Zhansaya_affectnet/train_set/train52k/Neutral/',
       r'/mnt/nas4/Zhansaya_affectnet/train_set/train52k/Happy/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train52k/Sad/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train52k/Surprise/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train52k/Fear/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train52k/Disgust/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train52k/Anger/',
      r'/mnt/nas4/Zhansaya_affectnet/train_set/train52k/Contempt/']

for path in paths:
    dir_path= path
    print(path[48:],len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))


# Validation data set

# In[33]:


# assign directory
directory2 = '/mnt/nas1/Zhansaya_affectnet/val_set/annotations/'


# In[34]:


val_exp_list=[]
for filename in glob.iglob(f'{directory2}/*'):
    if filename[-7:] == 'exp.npy':
        val_exp_list.append(filename)


# In[35]:


val_exp_list


# In[37]:


val_exp_list[0][49:-8]


# In[38]:


val_expdict = {}
for file in val_exp_list:
    val_expdict.update({file[49:-8]: np.load(file,allow_pickle=True)})


# In[39]:


len(val_expdict)


# In[40]:


val_folder_path = "/mnt/nas1/Zhansaya_affectnet/val_set/images/"

val_images = [f for f in os.listdir(val_folder_path) if os.path.isfile(os.path.join(val_folder_path, f))]


# In[41]:


for image in val_images:
    img_id = image.split('.')[0]
    for image_id, emotion_id in val_expdict.items():
        if img_id==image_id:
            emotion_id = emotion_id.astype(int)
            if emotion_id == 0:
                f_name='Neutral'
                new_path2 = os.path.join(val_folder_path, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(val_folder_path, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 1:
                f_name='Happy'
                new_path2 = os.path.join(val_folder_path, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(val_folder_path, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 2:
                f_name='Sad'
                new_path2 = os.path.join(val_folder_path, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(val_folder_path, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 3:
                f_name='Surprise'
                new_path2 = os.path.join(val_folder_path, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(val_folder_path, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 4:
                f_name='Fear'
                new_path2 = os.path.join(val_folder_path, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(val_folder_path, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 5:
                f_name='Disgust'
                new_path2 = os.path.join(val_folder_path, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(val_folder_path, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 6:
                f_name='Anger'
                new_path2 = os.path.join(val_folder_path, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(val_folder_path, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)
                
            if emotion_id == 7:
                f_name='Contempt'
                new_path2 = os.path.join(val_folder_path, f_name)
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                old_image_path = os.path.join(val_folder_path, image)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(old_image_path, new_image_path)

