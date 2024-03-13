#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
traindf=pd.read_csv('/home/snake/Desktop/imbalanced/FINALDF_idnames.csv')


# In[18]:


traindf


# In[19]:


traindf=traindf.drop(['Unnamed: 0'], axis=1)


# In[20]:


traindf


# In[21]:


b=[]
b1=[]
b2=[]
b3=[]
b4=[]
b5=[]
b6=[]
b7=[]
for i in traindf['Emotion']:
    if i=='Anger':
        b.append(i)
    elif i=='Contempt':
        b1.append(i)
    elif i=='Disgust':
        b2.append(i)
    elif i=='Fear':
        b3.append(i)
    elif i=='Happy':
        b4.append(i)
    elif i=='Neutral':
        b5.append(i)
    elif i=='Sad':
        b6.append(i)
    elif i=='Surprise':
        b7.append(i)
        


# In[22]:


print('Anger:',len(b))
print('Contempt:',len(b1))
print('Disgust:',len(b2))
print('Fear:',len(b3))
print('Happy:',len(b4))
print('Neutral:',len(b5))
print('Sad:',len(b6))
print('Surprise:',len(b7))


# In[23]:


anger_forward=[]
anger_left=[]
anger_right=[]
anger_up=[]
anger_down=[]
for i in range(len(traindf['Emotion'])):
    if traindf['Emotion'][i]=='Anger':
        if traindf['Direction'][i]==0:
            anger_forward.append(traindf['Filename'][i])
        if traindf['Direction'][i]==1:
            anger_left.append(traindf['Filename'][i])
        if traindf['Direction'][i]==2:
            anger_right.append(traindf['Filename'][i])
        if traindf['Direction'][i]==3:
            anger_up.append(traindf['Filename'][i])
        if traindf['Direction'][i]==4:
            anger_down.append(traindf['Filename'][i])


# In[24]:


print('Forward:',len(anger_forward))
print('Left:',len(anger_left))
print('Right:',len(anger_right))
print('Up:',len(anger_up))
print('Down:',len(anger_down))


# In[25]:


contempt_forward=[]
contempt_left=[]
contempt_right=[]
contempt_up=[]
contempt_down=[]
for i in range(len(traindf['Emotion'])):
    if traindf['Emotion'][i]=='Contempt':
        if traindf['Direction'][i]==0:
            contempt_forward.append(traindf['Filename'][i])
        if traindf['Direction'][i]==1:
            contempt_left.append(traindf['Filename'][i])
        if traindf['Direction'][i]==2:
            contempt_right.append(traindf['Filename'][i])
        if traindf['Direction'][i]==3:
            contempt_up.append(traindf['Filename'][i])
        if traindf['Direction'][i]==4:
            contempt_down.append(traindf['Filename'][i])
            
print('Forward:',len(contempt_forward))
print('Left:',len(contempt_left))
print('Right:',len(contempt_right))
print('Up:',len(contempt_up))
print('Down:',len(contempt_down))


# In[26]:


disgust_forward=[]
disgust_left=[]
disgust_right=[]
disgust_up=[]
disgust_down=[]
for i in range(len(traindf['Emotion'])):
    if traindf['Emotion'][i]=='Disgust':
        if traindf['Direction'][i]==0:
            disgust_forward.append(traindf['Filename'][i])
        if traindf['Direction'][i]==1:
            disgust_left.append(traindf['Filename'][i])
        if traindf['Direction'][i]==2:
            disgust_right.append(traindf['Filename'][i])
        if traindf['Direction'][i]==3:
            disgust_up.append(traindf['Filename'][i])
        if traindf['Direction'][i]==4:
            disgust_down.append(traindf['Filename'][i])
            
print('Forward:',len(disgust_forward))
print('Left:',len(disgust_left))
print('Right:',len(disgust_right))
print('Up:',len(disgust_up))
print('Down:',len(disgust_down))


# In[27]:


fear_forward=[]
fear_left=[]
fear_right=[]
fear_up=[]
fear_down=[]
for i in range(len(traindf['Emotion'])):
    if traindf['Emotion'][i]=='Fear':
        if traindf['Direction'][i]==0:
            fear_forward.append(traindf['Filename'][i])
        if traindf['Direction'][i]==1:
            fear_left.append(traindf['Filename'][i])
        if traindf['Direction'][i]==2:
            fear_right.append(traindf['Filename'][i])
        if traindf['Direction'][i]==3:
            fear_up.append(traindf['Filename'][i])
        if traindf['Direction'][i]==4:
            fear_down.append(traindf['Filename'][i])
            
print('Forward:',len(fear_forward))
print('Left:',len(fear_left))
print('Right:',len(fear_right))
print('Up:',len(fear_up))
print('Down:',len(fear_down))


# In[28]:


happy_forward=[]
happy_left=[]
happy_right=[]
happy_up=[]
happy_down=[]
for i in range(len(traindf['Emotion'])):
    if traindf['Emotion'][i]=='Happy':
        if traindf['Direction'][i]==0:
            happy_forward.append(traindf['Filename'][i])
        if traindf['Direction'][i]==1:
            happy_left.append(traindf['Filename'][i])
        if traindf['Direction'][i]==2:
            happy_right.append(traindf['Filename'][i])
        if traindf['Direction'][i]==3:
            happy_up.append(traindf['Filename'][i])
        if traindf['Direction'][i]==4:
            happy_down.append(traindf['Filename'][i])
            
print('Forward:',len(happy_forward))
print('Left:',len(happy_left))
print('Right:',len(happy_right))
print('Up:',len(happy_up))
print('Down:',len(happy_down))


# In[29]:


neutral_forward=[]
neutral_left=[]
neutral_right=[]
neutral_up=[]
neutral_down=[]
for i in range(len(traindf['Emotion'])):
    if traindf['Emotion'][i]=='Neutral':
        if traindf['Direction'][i]==0:
            neutral_forward.append(traindf['Filename'][i])
        if traindf['Direction'][i]==1:
            neutral_left.append(traindf['Filename'][i])
        if traindf['Direction'][i]==2:
            neutral_right.append(traindf['Filename'][i])
        if traindf['Direction'][i]==3:
            neutral_up.append(traindf['Filename'][i])
        if traindf['Direction'][i]==4:
            neutral_down.append(traindf['Filename'][i])
            
print('Forward:',len(neutral_forward))
print('Left:',len(neutral_left))
print('Right:',len(neutral_right))
print('Up:',len(neutral_up))
print('Down:',len(neutral_down))


# In[30]:


sad_forward=[]
sad_left=[]
sad_right=[]
sad_up=[]
sad_down=[]
for i in range(len(traindf['Emotion'])):
    if traindf['Emotion'][i]=='Sad':
        if traindf['Direction'][i]==0:
            sad_forward.append(traindf['Filename'][i])
        if traindf['Direction'][i]==1:
            sad_left.append(traindf['Filename'][i])
        if traindf['Direction'][i]==2:
            sad_right.append(traindf['Filename'][i])
        if traindf['Direction'][i]==3:
            sad_up.append(traindf['Filename'][i])
        if traindf['Direction'][i]==4:
            sad_down.append(traindf['Filename'][i])
            
print('Forward:',len(sad_forward))
print('Left:',len(sad_left))
print('Right:',len(sad_right))
print('Up:',len(sad_up))
print('Down:',len(sad_down))


# In[31]:


surprise_forward=[]
surprise_left=[]
surprise_right=[]
surprise_up=[]
surprise_down=[]
for i in range(len(traindf['Emotion'])):
    if traindf['Emotion'][i]=='Surprise':
        if traindf['Direction'][i]==0:
            surprise_forward.append(traindf['Filename'][i])
        if traindf['Direction'][i]==1:
            surprise_left.append(traindf['Filename'][i])
        if traindf['Direction'][i]==2:
            surprise_right.append(traindf['Filename'][i])
        if traindf['Direction'][i]==3:
            surprise_up.append(traindf['Filename'][i])
        if traindf['Direction'][i]==4:
            surprise_down.append(traindf['Filename'][i])
            
print('Forward:',len(surprise_forward))
print('Left:',len(surprise_left))
print('Right:',len(surprise_right))
print('Up:',len(surprise_up))
print('Down:',len(surprise_down))


# In[36]:


balanced_anger_forward=[]
balanced_anger_left=[]
balanced_anger_right=[]
balanced_anger_up=[]
balanced_anger_down=[]

balanced_contempt_forward=[]
balanced_contempt_left=[]
balanced_contempt_right=[]
balanced_contempt_up=[]
balanced_contempt_down=[]

balanced_disgust_forward=[]
balanced_disgust_left=[]
balanced_disgust_right=[]
balanced_disgust_up=[]
balanced_disgust_down=[]

balanced_fear_forward=[]
balanced_fear_left=[]
balanced_fear_right=[]
balanced_fear_up=[]
balanced_fear_down=[]

balanced_happy_forward=[]
balanced_happy_left=[]
balanced_happy_right=[]
balanced_happy_up=[]
balanced_happy_down=[]

balanced_neutral_forward=[]
balanced_neutral_left=[]
balanced_neutral_right=[]
balanced_neutral_up=[]
balanced_neutral_down=[]

balanced_sad_forward=[]
balanced_sad_left=[]
balanced_sad_right=[]
balanced_sad_up=[]
balanced_sad_down=[]

balanced_surprise_forward=[]
balanced_surprise_left=[]
balanced_surprise_right=[]
balanced_surprise_up=[]
balanced_surprise_down=[]


# In[37]:


for i in range(251):
    balanced_anger_forward.append(anger_forward[i])
    balanced_anger_left.append(anger_left[i])
    balanced_anger_right.append(anger_right[i])
    balanced_anger_up.append(anger_up[i])
    balanced_anger_down.append(anger_down[i])
    
    balanced_contempt_forward.append(contempt_forward[i])
    balanced_contempt_left.append(contempt_left[i])
    balanced_contempt_right.append(contempt_right[i])
    balanced_contempt_up.append(contempt_up[i])
    balanced_contempt_down.append(contempt_down[i])
    
    balanced_disgust_forward.append(disgust_forward[i])
    balanced_disgust_left.append(disgust_left[i])
    balanced_disgust_right.append(disgust_right[i])
    balanced_disgust_up.append(disgust_up[i])
    balanced_disgust_down.append(disgust_down[i])
    
    balanced_fear_forward.append(fear_forward[i])
    balanced_fear_left.append(fear_left[i])
    balanced_fear_right.append(fear_right[i])
    balanced_fear_up.append(fear_up[i])
    balanced_fear_down.append(fear_down[i])
    
    balanced_happy_forward.append(happy_forward[i])
    balanced_happy_left.append(happy_left[i])
    balanced_happy_right.append(happy_right[i])
    balanced_happy_up.append(happy_up[i])
    balanced_happy_down.append(happy_down[i])
    
    balanced_neutral_forward.append(neutral_forward[i])
    balanced_neutral_left.append(neutral_left[i])
    balanced_neutral_right.append(neutral_right[i])
    balanced_neutral_up.append(neutral_up[i])
    balanced_neutral_down.append(neutral_down[i])
    
    balanced_sad_forward.append(sad_forward[i])
    balanced_sad_left.append(sad_left[i])
    balanced_sad_right.append(sad_right[i])
    balanced_sad_up.append(sad_up[i])
    balanced_sad_down.append(sad_down[i])
    
    balanced_surprise_forward.append(surprise_forward[i])
    balanced_surprise_left.append(surprise_left[i])
    balanced_surprise_right.append(surprise_right[i])
    balanced_surprise_up.append(surprise_up[i])
    balanced_surprise_down.append(surprise_down[i])


# In[53]:


rbalanced_anger_forward=[]
rbalanced_anger_left=[]
rbalanced_anger_right=[]
rbalanced_anger_up=[]
rbalanced_anger_down=[]

rbalanced_contempt_forward=[]
rbalanced_contempt_left=[]
rbalanced_contempt_right=[]
rbalanced_contempt_up=[]
rbalanced_contempt_down=[]

rbalanced_disgust_forward=[]
rbalanced_disgust_left=[]
rbalanced_disgust_right=[]
rbalanced_disgust_up=[]
rbalanced_disgust_down=[]

rbalanced_fear_forward=[]
rbalanced_fear_left=[]
rbalanced_fear_right=[]
rbalanced_fear_up=[]
rbalanced_fear_down=[]

rbalanced_happy_forward=[]
rbalanced_happy_left=[]
rbalanced_happy_right=[]
rbalanced_happy_up=[]
rbalanced_happy_down=[]

rbalanced_neutral_forward=[]
rbalanced_neutral_left=[]
rbalanced_neutral_right=[]
rbalanced_neutral_up=[]
rbalanced_neutral_down=[]

rbalanced_sad_forward=[]
rbalanced_sad_left=[]
rbalanced_sad_right=[]
rbalanced_sad_up=[]
rbalanced_sad_down=[]

rbalanced_surprise_forward=[]
rbalanced_surprise_left=[]
rbalanced_surprise_right=[]
rbalanced_surprise_up=[]
rbalanced_surprise_down=[]


# In[56]:


for i in range(252,262):
    rbalanced_anger_forward.append(anger_forward[i])
    rbalanced_anger_left.append(anger_left[i])
    rbalanced_anger_right.append(anger_right[i])
    rbalanced_anger_up.append(anger_up[i])
    rbalanced_anger_down.append(anger_down[i])
    
    rbalanced_contempt_forward.append(contempt_forward[i])
    
    
    rbalanced_contempt_up.append(contempt_up[i])
    
    
    rbalanced_disgust_forward.append(disgust_forward[i])
    rbalanced_disgust_left.append(disgust_left[i])
    rbalanced_disgust_right.append(disgust_right[i])
    rbalanced_disgust_up.append(disgust_up[i])
    rbalanced_disgust_down.append(disgust_down[i])
    
    rbalanced_fear_forward.append(fear_forward[i])
    rbalanced_fear_left.append(fear_left[i])
    rbalanced_fear_right.append(fear_right[i])
    rbalanced_fear_up.append(fear_up[i])
    rbalanced_fear_down.append(fear_down[i])
    
    rbalanced_happy_forward.append(happy_forward[i])
    rbalanced_happy_left.append(happy_left[i])
    rbalanced_happy_right.append(happy_right[i])
    rbalanced_happy_up.append(happy_up[i])
    rbalanced_happy_down.append(happy_down[i])
    
    rbalanced_neutral_forward.append(neutral_forward[i])
    rbalanced_neutral_left.append(neutral_left[i])
    rbalanced_neutral_right.append(neutral_right[i])
    rbalanced_neutral_up.append(neutral_up[i])
    rbalanced_neutral_down.append(neutral_down[i])
    
    rbalanced_sad_forward.append(sad_forward[i])
    rbalanced_sad_left.append(sad_left[i])
    rbalanced_sad_right.append(sad_right[i])
    rbalanced_sad_up.append(sad_up[i])
    rbalanced_sad_down.append(sad_down[i])
    
    rbalanced_surprise_forward.append(surprise_forward[i])
    rbalanced_surprise_left.append(surprise_left[i])
    
    rbalanced_surprise_up.append(surprise_up[i])
    rbalanced_surprise_down.append(surprise_down[i])


# In[57]:


print(rbalanced_anger_forward)


# In[58]:


print(rbalanced_anger_down)


# In[59]:


print(rbalanced_anger_left)


# In[60]:


print(rbalanced_anger_right)


# In[61]:


print(rbalanced_contempt_forward)


# In[62]:


print(rbalanced_contempt_up)


# In[63]:


print(rbalanced_disgust_forward)


# In[64]:


print(rbalanced_disgust_down)


# In[65]:


print(rbalanced_disgust_left)


# In[66]:


print(rbalanced_disgust_right)


# In[67]:


print(rbalanced_disgust_up)


# In[68]:


print(rbalanced_fear_down)


# In[69]:


print(rbalanced_fear_forward)


# In[70]:


print(rbalanced_fear_left)


# In[71]:


print(rbalanced_fear_right)


# In[72]:


print(rbalanced_fear_up)


# In[73]:


print(rbalanced_happy_forward)


# In[74]:


print(rbalanced_happy_left)


# In[75]:


print(rbalanced_happy_right)


# In[76]:


print(rbalanced_happy_up)


# In[77]:


print(rbalanced_happy_down)


# In[78]:


print(rbalanced_neutral_forward)


# In[79]:


print(rbalanced_neutral_left)


# In[80]:


print(rbalanced_neutral_right)


# In[81]:


print(rbalanced_neutral_up)


# In[82]:


print(rbalanced_neutral_down)


# In[83]:


print(rbalanced_sad_forward)


# In[84]:


print(rbalanced_sad_left)


# In[85]:


print(rbalanced_sad_right)


# In[86]:


print(rbalanced_sad_up)


# In[87]:


print(rbalanced_sad_down)


# In[88]:


print(rbalanced_surprise_forward)


# In[89]:


print(rbalanced_surprise_left)


# In[91]:


print(rbalanced_surprise_up)


# In[92]:


print(rbalanced_surprise_down)


# In[41]:


import os
#Put all images in one folder
anger_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/FINAL_DATASET/Anger/"
anger_images = [f for f in os.listdir(anger_folder_path) if os.path.isfile(os.path.join(anger_folder_path, f))]

#Put all images in one folder
contempt_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/FINAL_DATASET/Contempt/"
contempt_images = [f for f in os.listdir(contempt_folder_path) if os.path.isfile(os.path.join(contempt_folder_path, f))]

#Put all images in one folder
disgust_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/FINAL_DATASET/Disgust/"
disgust_images = [f for f in os.listdir(disgust_folder_path) if os.path.isfile(os.path.join(disgust_folder_path, f))]

#Put all images in one folder
fear_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/FINAL_DATASET/Fear/"
fear_images = [f for f in os.listdir(fear_folder_path) if os.path.isfile(os.path.join(fear_folder_path, f))]

#Put all images in one folder
happy_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/FINAL_DATASET/Happy/"
happy_images = [f for f in os.listdir(happy_folder_path) if os.path.isfile(os.path.join(happy_folder_path, f))]

#Put all images in one folder
neutral_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/FINAL_DATASET/Neutral/"
neutral_images = [f for f in os.listdir(neutral_folder_path) if os.path.isfile(os.path.join(neutral_folder_path, f))]

#Put all images in one folder
sad_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/FINAL_DATASET/Sad/"
sad_images = [f for f in os.listdir(sad_folder_path) if os.path.isfile(os.path.join(sad_folder_path, f))]

#Put all images in one folder
surprise_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/FINAL_DATASET/Surprise/"
surprise_images = [f for f in os.listdir(surprise_folder_path) if os.path.isfile(os.path.join(surprise_folder_path, f))]


# In[43]:


import os, os.path, shutil

bad_forward_anger=[]

new_path1= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Anger/Forward/'
for i in range(len(balanced_anger_forward)):
    try:
        old_image_path = os.path.join(anger_folder_path, balanced_anger_forward[i])
        new_image_path = os.path.join(new_path1, balanced_anger_forward[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_forward_anger.append(balanced_anger_forward[i])


# In[44]:


print(len(bad_forward_anger))


# In[45]:


bad_left_anger=[]   
new_path2= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Anger/Left/'  
for i in range(len(balanced_anger_left)):
    try:
        old_image_path = os.path.join(anger_folder_path, balanced_anger_left[i])
        new_image_path = os.path.join(new_path2, balanced_anger_left[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_left_anger.append(balanced_anger_forward[i])
        
print(len(bad_left_anger))

bad_right_anger=[]  
new_path3= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Anger/Right/'    
for i in range(len(balanced_anger_right)):
    try:
        old_image_path = os.path.join(anger_folder_path, balanced_anger_right[i])
        new_image_path = os.path.join(new_path3, balanced_anger_right[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_right_anger.append(balanced_anger_right[i])
        
print(len(bad_right_anger))

bad_up_anger=[]
new_path4= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Anger/Up/'
for i in range(len(balanced_anger_up)):
    try:
        old_image_path = os.path.join(anger_folder_path, balanced_anger_up[i])
        new_image_path = os.path.join(new_path4, balanced_anger_up[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_up_anger.append(balanced_anger_up[i])

print(len(bad_up_anger))


bad_down_anger=[]
new_path5= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Anger/Down/'
for i in range(len(balanced_anger_down)):
    try:
        old_image_path = os.path.join(anger_folder_path, balanced_anger_down[i])
        new_image_path = os.path.join(new_path5, balanced_anger_down[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_down_anger.append(balanced_anger_down[i])

print(len(bad_down_anger))


# In[46]:


bad_forward_contempt=[]

new_path1= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Contempt/Forward/'
for i in range(len(balanced_contempt_forward)):
    try:
        old_image_path = os.path.join(contempt_folder_path, balanced_contempt_forward[i])
        new_image_path = os.path.join(new_path1, balanced_contempt_forward[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_forward_contempt.append(balanced_contempt_forward[i])
        
bad_left_contempt=[]   
new_path2= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Contempt/Left/'  
for i in range(len(balanced_contempt_left)):
    try:
        old_image_path = os.path.join(contempt_folder_path, balanced_contempt_left[i])
        new_image_path = os.path.join(new_path2, balanced_contempt_left[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_left_contempt.append(balanced_contempt_forward[i])
        
print(len(bad_left_contempt))

bad_right_contempt=[]  
new_path3= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Contempt/Right/'    
for i in range(len(balanced_contempt_right)):
    try:
        old_image_path = os.path.join(contempt_folder_path, balanced_contempt_right[i])
        new_image_path = os.path.join(new_path3, balanced_contempt_right[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_right_contempt.append(balanced_contempt_right[i])
        
print(len(bad_right_contempt))

bad_up_contempt=[]
new_path4= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Contempt/Up/'
for i in range(len(balanced_contempt_up)):
    try:
        old_image_path = os.path.join(contempt_folder_path, balanced_contempt_up[i])
        new_image_path = os.path.join(new_path4, balanced_contempt_up[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_up_contempt.append(balanced_contempt_up[i])

print(len(bad_up_contempt))


bad_down_contempt=[]
new_path5= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Contempt/Down/'
for i in range(len(balanced_contempt_down)):
    try:
        old_image_path = os.path.join(contempt_folder_path, balanced_contempt_down[i])
        new_image_path = os.path.join(new_path5, balanced_contempt_down[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_down_contempt.append(balanced_contempt_down[i])

print(len(bad_down_contempt))        


# In[47]:


bad_forward_disgust=[]

new_path1= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Disgust/Forward/'
for i in range(len(balanced_disgust_forward)):
    try:
        old_image_path = os.path.join(disgust_folder_path, balanced_disgust_forward[i])
        new_image_path = os.path.join(new_path1, balanced_disgust_forward[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_forward_disgust.append(balanced_disgust_forward[i])

print(len(bad_forward_disgust))

bad_left_disgust=[]   
new_path2= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Disgust/Left/'  
for i in range(len(balanced_disgust_left)):
    try:
        old_image_path = os.path.join(disgust_folder_path, balanced_disgust_left[i])
        new_image_path = os.path.join(new_path2, balanced_disgust_left[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_left_disgust.append(balanced_disgust_forward[i])
        
print(len(bad_left_disgust))

bad_right_disgust=[]  
new_path3= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Disgust/Right/'    
for i in range(len(balanced_disgust_right)):
    try:
        old_image_path = os.path.join(disgust_folder_path, balanced_disgust_right[i])
        new_image_path = os.path.join(new_path3, balanced_disgust_right[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_right_disgust.append(balanced_disgust_right[i])
        
print(len(bad_right_disgust))

bad_up_disgust=[]
new_path4= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Disgust/Up/'
for i in range(len(balanced_disgust_up)):
    try:
        old_image_path = os.path.join(disgust_folder_path, balanced_disgust_up[i])
        new_image_path = os.path.join(new_path4, balanced_disgust_up[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_up_disgust.append(balanced_disgust_up[i])

print(len(bad_up_disgust))


bad_down_disgust=[]
new_path5= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Disgust/Down/'
for i in range(len(balanced_disgust_down)):
    try:
        old_image_path = os.path.join(disgust_folder_path, balanced_disgust_down[i])
        new_image_path = os.path.join(new_path5, balanced_disgust_down[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_down_disgust.append(balanced_disgust_down[i])

print(len(bad_down_disgust))        


# In[48]:


bad_forward_fear=[]

new_path1= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Fear/Forward/'
for i in range(len(balanced_fear_forward)):
    try:
        old_image_path = os.path.join(fear_folder_path, balanced_fear_forward[i])
        new_image_path = os.path.join(new_path1, balanced_fear_forward[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_forward_fear.append(balanced_fear_forward[i])

print(len(bad_forward_fear))

bad_left_fear=[]   
new_path2= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Fear/Left/'  
for i in range(len(balanced_fear_left)):
    try:
        old_image_path = os.path.join(fear_folder_path, balanced_fear_left[i])
        new_image_path = os.path.join(new_path2, balanced_fear_left[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_left_fear.append(balanced_fear_forward[i])
        
print(len(bad_left_fear))

bad_right_fear=[]  
new_path3= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Fear/Right/'    
for i in range(len(balanced_fear_right)):
    try:
        old_image_path = os.path.join(fear_folder_path, balanced_fear_right[i])
        new_image_path = os.path.join(new_path3, balanced_fear_right[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_right_fear.append(balanced_fear_right[i])
        
print(len(bad_right_fear))

bad_up_fear=[]
new_path4= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Fear/Up/'
for i in range(len(balanced_fear_up)):
    try:
        old_image_path = os.path.join(fear_folder_path, balanced_fear_up[i])
        new_image_path = os.path.join(new_path4, balanced_fear_up[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_up_fear.append(balanced_fear_up[i])

print(len(bad_up_fear))


bad_down_fear=[]
new_path5= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Fear/Down/'
for i in range(len(balanced_fear_down)):
    try:
        old_image_path = os.path.join(fear_folder_path, balanced_fear_down[i])
        new_image_path = os.path.join(new_path5, balanced_fear_down[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_down_fear.append(balanced_fear_down[i])

print(len(bad_down_fear))   


# In[49]:


bad_forward_happy=[]

new_path1= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Happy/Forward/'
for i in range(len(balanced_happy_forward)):
    try:
        old_image_path = os.path.join(happy_folder_path, balanced_happy_forward[i])
        new_image_path = os.path.join(new_path1, balanced_happy_forward[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_forward_happy.append(balanced_happy_forward[i])

print(len(bad_forward_happy))

bad_left_happy=[]   
new_path2= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Happy/Left/'  
for i in range(len(balanced_happy_left)):
    try:
        old_image_path = os.path.join(happy_folder_path, balanced_happy_left[i])
        new_image_path = os.path.join(new_path2, balanced_happy_left[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_left_happy.append(balanced_happy_forward[i])
        
print(len(bad_left_happy))

bad_right_happy=[]  
new_path3= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Happy/Right/'    
for i in range(len(balanced_happy_right)):
    try:
        old_image_path = os.path.join(happy_folder_path, balanced_happy_right[i])
        new_image_path = os.path.join(new_path3, balanced_happy_right[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_right_happy.append(balanced_happy_right[i])
        
print(len(bad_right_happy))

bad_up_happy=[]
new_path4= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Happy/Up/'
for i in range(len(balanced_happy_up)):
    try:
        old_image_path = os.path.join(happy_folder_path, balanced_happy_up[i])
        new_image_path = os.path.join(new_path4, balanced_happy_up[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_up_happy.append(balanced_happy_up[i])

print(len(bad_up_happy))


bad_down_happy=[]
new_path5= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Happy/Down/'
for i in range(len(balanced_happy_down)):
    try:
        old_image_path = os.path.join(happy_folder_path, balanced_happy_down[i])
        new_image_path = os.path.join(new_path5, balanced_happy_down[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_down_happy.append(balanced_happy_down[i])

print(len(bad_down_happy))   


# In[50]:


bad_forward_neutral=[]

new_path1= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Neutral/Forward/'
for i in range(len(balanced_neutral_forward)):
    try:
        old_image_path = os.path.join(neutral_folder_path, balanced_neutral_forward[i])
        new_image_path = os.path.join(new_path1, balanced_neutral_forward[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_forward_neutral.append(balanced_neutral_forward[i])

print(len(bad_forward_neutral))

bad_left_neutral=[]   
new_path2= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Neutral/Left/'  
for i in range(len(balanced_neutral_left)):
    try:
        old_image_path = os.path.join(neutral_folder_path, balanced_neutral_left[i])
        new_image_path = os.path.join(new_path2, balanced_neutral_left[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_left_neutral.append(balanced_neutral_forward[i])
        
print(len(bad_left_neutral))

bad_right_neutral=[]  
new_path3= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Neutral/Right/'    
for i in range(len(balanced_neutral_right)):
    try:
        old_image_path = os.path.join(neutral_folder_path, balanced_neutral_right[i])
        new_image_path = os.path.join(new_path3, balanced_neutral_right[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_right_neutral.append(balanced_neutral_right[i])
        
print(len(bad_right_neutral))

bad_up_neutral=[]
new_path4= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Neutral/Up/'
for i in range(len(balanced_neutral_up)):
    try:
        old_image_path = os.path.join(neutral_folder_path, balanced_neutral_up[i])
        new_image_path = os.path.join(new_path4, balanced_neutral_up[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_up_neutral.append(balanced_neutral_up[i])

print(len(bad_up_neutral))


bad_down_neutral=[]
new_path5= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Neutral/Down/'
for i in range(len(balanced_neutral_down)):
    try:
        old_image_path = os.path.join(neutral_folder_path, balanced_neutral_down[i])
        new_image_path = os.path.join(new_path5, balanced_neutral_down[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_down_neutral.append(balanced_neutral_down[i])

print(len(bad_down_neutral)) 


# In[51]:


bad_forward_sad=[]

new_path1= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Sad/Forward/'
for i in range(len(balanced_sad_forward)):
    try:
        old_image_path = os.path.join(sad_folder_path, balanced_sad_forward[i])
        new_image_path = os.path.join(new_path1, balanced_sad_forward[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_forward_sad.append(balanced_sad_forward[i])

print(len(bad_forward_sad))

bad_left_sad=[]   
new_path2= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Sad/Left/'  
for i in range(len(balanced_sad_left)):
    try:
        old_image_path = os.path.join(sad_folder_path, balanced_sad_left[i])
        new_image_path = os.path.join(new_path2, balanced_sad_left[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_left_sad.append(balanced_sad_forward[i])
        
print(len(bad_left_sad))

bad_right_sad=[]  
new_path3= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Sad/Right/'    
for i in range(len(balanced_sad_right)):
    try:
        old_image_path = os.path.join(sad_folder_path, balanced_sad_right[i])
        new_image_path = os.path.join(new_path3, balanced_sad_right[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_right_sad.append(balanced_sad_right[i])
        
print(len(bad_right_sad))

bad_up_sad=[]
new_path4= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Sad/Up/'
for i in range(len(balanced_sad_up)):
    try:
        old_image_path = os.path.join(sad_folder_path, balanced_sad_up[i])
        new_image_path = os.path.join(new_path4, balanced_sad_up[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_up_sad.append(balanced_sad_up[i])

print(len(bad_up_sad))


bad_down_sad=[]
new_path5= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Sad/Down/'
for i in range(len(balanced_sad_down)):
    try:
        old_image_path = os.path.join(sad_folder_path, balanced_sad_down[i])
        new_image_path = os.path.join(new_path5, balanced_sad_down[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_down_sad.append(balanced_sad_down[i])

print(len(bad_down_sad))


# In[52]:


bad_forward_surprise=[]

new_path1= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Surprise/Forward/'
for i in range(len(balanced_surprise_forward)):
    try:
        old_image_path = os.path.join(surprise_folder_path, balanced_surprise_forward[i])
        new_image_path = os.path.join(new_path1, balanced_surprise_forward[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_forward_surprise.append(balanced_surprise_forward[i])

print(len(bad_forward_surprise))

bad_left_surprise=[]   
new_path2= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Surprise/Left/'  
for i in range(len(balanced_surprise_left)):
    try:
        old_image_path = os.path.join(surprise_folder_path, balanced_surprise_left[i])
        new_image_path = os.path.join(new_path2, balanced_surprise_left[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_left_surprise.append(balanced_surprise_forward[i])
        
print(len(bad_left_surprise))

bad_right_surprise=[]  
new_path3= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Surprise/Right/'    
for i in range(len(balanced_surprise_right)):
    try:
        old_image_path = os.path.join(surprise_folder_path, balanced_surprise_right[i])
        new_image_path = os.path.join(new_path3, balanced_surprise_right[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_right_sad.append(balanced_surprise_right[i])
        
print(len(bad_right_surprise))

bad_up_surprise=[]
new_path4= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Surprise/Up/'
for i in range(len(balanced_surprise_up)):
    try:
        old_image_path = os.path.join(surprise_folder_path, balanced_surprise_up[i])
        new_image_path = os.path.join(new_path4, balanced_surprise_up[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_up_surprise.append(balanced_surprise_up[i])

print(len(bad_up_surprise))


bad_down_surprise=[]
new_path5= '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/Surprise/Down/'
for i in range(len(balanced_surprise_down)):
    try:
        old_image_path = os.path.join(surprise_folder_path, balanced_surprise_down[i])
        new_image_path = os.path.join(new_path5, balanced_surprise_down[i])
        shutil.move(old_image_path, new_image_path)
    except FileNotFoundError:
        #print("FileNotFoundError: "+ balanced_anger_forward[i])
        bad_down_surprise.append(balanced_surprise_down[i])

print(len(bad_down_surprise))


# In[ ]:




