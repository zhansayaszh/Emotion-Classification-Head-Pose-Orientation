#!/usr/bin/env python
# coding: utf-8

# In[1]:


def get_pair(line):
    key, sep, value = line.strip().partition(":")
    return key, value


# In[2]:


anger_path = "/home/snake/Desktop/zhansaya/new_emotion_dictionaries/anger.txt"
contempt_path="/home/snake/Desktop/zhansaya/new_emotion_dictionaries/contempt.txt"
disgust_path="/home/snake/Desktop/zhansaya/new_emotion_dictionaries/disgust.txt"
fear_path="/home/snake/Desktop/zhansaya/new_emotion_dictionaries/fear.txt"
happy_path="/home/snake/Desktop/zhansaya/new_emotion_dictionaries/happy.txt"
neutral_path="/home/snake/Desktop/zhansaya/new_emotion_dictionaries/neutral.txt"
sad_path="/home/snake/Desktop/zhansaya/new_emotion_dictionaries/sad.txt"
surprise_path="/home/snake/Desktop/zhansaya/new_emotion_dictionaries/surprise.txt"


# In[3]:


with open(anger_path) as fd:    
    anger = dict(get_pair(line) for line in fd)

with open(contempt_path) as fd:    
    contempt = dict(get_pair(line) for line in fd)

with open(disgust_path) as fd:    
    disgust = dict(get_pair(line) for line in fd)

with open(fear_path) as fd:    
    fear = dict(get_pair(line) for line in fd)

with open(happy_path) as fd:    
    happy = dict(get_pair(line) for line in fd)

with open(neutral_path) as fd:    
    neutral = dict(get_pair(line) for line in fd)

with open(sad_path) as fd:    
    sad = dict(get_pair(line) for line in fd)

with open(surprise_path) as fd:    
    surprise = dict(get_pair(line) for line in fd)


# In[5]:


new_anger={}
for x, y in anger.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_anger[x]=z


# In[6]:


new_contempt={}
for x, y in contempt.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_contempt[x]=z


# In[7]:


new_disgust={}
for x, y in disgust.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_disgust[x]=z


# In[8]:


new_fear={}
for x, y in fear.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_fear[x]=z


# In[9]:


new_happy={}
for x, y in happy.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_happy[x]=z


# In[10]:


new_neutral={}
for x, y in neutral.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_neutral[x]=z


# In[11]:


new_sad={}
for x, y in sad.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_sad[x]=z


# In[12]:


new_surprise={}
for x, y in surprise.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_surprise[x]=z


# In[16]:


type(new_surprise['/mnt/nas4/Zhansaya_affectnet/train_set/imbalanced/Surprise/116158.jpg']


# In[17]:


import pandas as pd
test_df = pd.DataFrame(columns=['Filename', 'Pitch', 'Yaw', 'Roll'])


# In[18]:


for x, y in new_anger.items():
    test_df = test_df.append({'Filename': x, 'Pitch': y[0], 'Yaw': y[1],'Roll': y[2]}, ignore_index=True)


# In[19]:


for x, y in new_contempt.items():
    test_df = test_df.append({'Filename': x, 'Pitch': y[0], 'Yaw': y[1],'Roll': y[2]}, ignore_index=True)


# In[20]:


for x, y in new_disgust.items():
    test_df = test_df.append({'Filename': x, 'Pitch': y[0], 'Yaw': y[1],'Roll': y[2]}, ignore_index=True)


# In[21]:


for x, y in new_fear.items():
    test_df = test_df.append({'Filename': x, 'Pitch': y[0], 'Yaw': y[1],'Roll': y[2]}, ignore_index=True)


# In[22]:


for x, y in new_happy.items():
    test_df = test_df.append({'Filename': x, 'Pitch': y[0], 'Yaw': y[1],'Roll': y[2]}, ignore_index=True)


# In[23]:


for x, y in new_neutral.items():
    test_df = test_df.append({'Filename': x, 'Pitch': y[0], 'Yaw': y[1],'Roll': y[2]}, ignore_index=True)


# In[24]:


for x, y in new_sad.items():
    test_df = test_df.append({'Filename': x, 'Pitch': y[0], 'Yaw': y[1],'Roll': y[2]}, ignore_index=True)


# In[25]:


for x, y in new_surprise.items():
    test_df = test_df.append({'Filename': x, 'Pitch': y[0], 'Yaw': y[1],'Roll': y[2]}, ignore_index=True)


# In[29]:


test_df.to_csv('/home/snake/Desktop/imbalanced/last_testdf.csv')


# In[32]:


import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# In[33]:


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x


# In[37]:


import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define the model architecture
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(3, 5) # 8 input features, 5 output classes
        
    def forward(self, x):
        x = self.fc1(x)
        return x

# Load the data
data = pd.read_csv('/home/snake/Desktop/imbalanced/last_testdf.csv')

# Split X and y
X = data.iloc[:, 2:-1].values # take columns 2 through second-to-last
y = data.iloc[:, -1].values # take last column

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = LabelEncoder().fit_transform(y)

# Load the trained model
model = MyModel()
# Load the state_dict
state_dict = torch.load('/home/snake/Desktop/imbalanced/nnmodel.pth')

# Create a new instance of the model
model = MulticlassClassification(3, 5)

state_dict = {k.replace('layer_layer_out', 'layer_out'): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)


# Use the model to predict direction
with torch.no_grad():
    X = torch.Tensor(X)
    predictions = model(X)
    predicted_classes = predictions.argmax(dim=1)

# Interpret the predictions
direction_labels = ['0', '1', '2', '3', '4']
predicted_directions = [direction_labels[prediction] for prediction in predicted_classes]


# In[39]:


# Create a new column with the predicted directions
data['Direction'] = predicted_directions

# Save the DataFrame to a CSV file
data.to_csv('/home/snake/Desktop/imbalanced/last_testdf_predicted.csv', index=False)


# In[40]:


num=0
for i in predicted_directions:
  if i=='0':
    num=num+1

print(num)


# In[41]:


num=0
for i in predicted_directions:
  if i=='1':
    num=num+1

print(num)


# In[42]:


num=0
for i in predicted_directions:
  if i=='2':
    num=num+1

print(num)


# In[43]:


num=0
for i in predicted_directions:
  if i=='3':
    num=num+1

print(num)


# In[44]:


num=0
for i in predicted_directions:
  if i=='4':
    num=num+1

print(num)


# In[65]:


alldf = pd.read_csv('/home/snake/Desktop/imbalanced/last_testdf_predicted.csv')


# In[50]:


alldf['Filename'][0].split('/')[-2]


# In[51]:


len(alldf)


# In[67]:


alldf["Emotion"] = None


# In[68]:


alldf.head()


# In[76]:


alldf


# In[69]:


emotions=[]
for i in alldf['Filename']:
    emotion=i.split('/')[-2]
    emotions.append(emotion)


# In[73]:


num=0
for i in emotions:
    if i=='Anger':
        num=num+1


# In[74]:


num


# In[75]:


len(emotions)


# In[77]:


alldf['Emotion']=emotions


# In[78]:


alldf


# In[79]:


alldf.to_csv('/home/snake/Desktop/imbalanced/lastdf_with_emotions.csv')


# In[80]:


alldf2 = pd.read_csv('/home/snake/Desktop/imbalanced/test_directions_predicted.csv')


# In[87]:


alldf2['Filename'][0].split('/')[-2]


# In[88]:


alldf2["Emotion"] = None


# In[89]:


emotions2=[]
for i in alldf2['Filename']:
    emotion=i.split('/')[-2]
    emotions2.append(emotion)


# In[90]:


alldf2['Emotion']=emotions2


# In[91]:


alldf2


# In[93]:


df=pd.concat([alldf, alldf2], axis=0)


# In[94]:


df.to_csv('/home/snake/Desktop/imbalanced/alld')


# In[ ]:




