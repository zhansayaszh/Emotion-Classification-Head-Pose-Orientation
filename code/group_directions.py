#!/usr/bin/env python
# coding: utf-8

# In[114]:


def get_pair(line):
    key, sep, value = line.strip().partition(":")
    return key, value


# In[115]:


anger_path = "/home/snake/Desktop/zhansaya/emotion_dictionaries/anger.txt"
contempt_path="/home/snake/Desktop/zhansaya/emotion_dictionaries/contempt.txt"
disgust_path="/home/snake/Desktop/zhansaya/emotion_dictionaries/disgust.txt"
fear_path="/home/snake/Desktop/zhansaya/emotion_dictionaries/fear.txt"
happy_path="/home/snake/Desktop/zhansaya/emotion_dictionaries/happy.txt"
neutral_path="/home/snake/Desktop/zhansaya/emotion_dictionaries/neutral.txt"
sad_path="/home/snake/Desktop/zhansaya/emotion_dictionaries/sad.txt"
surprise_path="/home/snake/Desktop/zhansaya/emotion_dictionaries/surprise.txt"


# In[116]:


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


# In[117]:


new_anger={}
for x, y in anger.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_anger[x]=z


# In[118]:


new_contempt={}
for x, y in contempt.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_contempt[x]=z


# In[119]:


new_disgust={}
for x, y in disgust.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_disgust[x]=z


# In[120]:


new_fear={}
for x, y in fear.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_fear[x]=z


# In[121]:


new_happy={}
for x, y in happy.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_happy[x]=z


# In[122]:


new_neutral={}
for x, y in neutral.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_neutral[x]=z


# In[123]:


new_sad={}
for x, y in sad.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_sad[x]=z


# In[124]:


new_surprise={}
for x, y in surprise.items():
  y = y.replace("tensor", "")
  y = y.replace("(", "")
  y = y.replace(")", "")
  y = y.replace(",", " ")
  z = y.split()
  z = list(map(float, z))
  new_surprise[x]=z


# In[125]:


path_dir_anger={}
for x, y in new_anger.items():
  #print("pitch:",y[0],"yaw:",y[1],"roll:",y[2])
  if y[1]<y[0] and y[1]<y[2] and y[1]<0 and y[1]<-15:
    if y[0]>0 and y[2]<0:
      if y[0]<3:
        path_dir_anger[x]="left"
      else:
        path_dir_anger[x]="up"
    elif y[0]<0 and y[2]<0:
      if y[0]+y[2]>-15:
        path_dir_anger[x]="left"
      else:
        path_dir_anger[x]="down"
    else:
      path_dir_anger[x]="left"
  elif y[1]>y[0] and y[1]>y[2] and y[1]>0 and y[1]>5:
    if y[0]+y[2]>-9 and y[0]<0 and y[2]<0:
      path_dir_anger[x]="right"
    else:
      if y[0]+y[2]>0:
        path_dir_anger[x]="up"
      elif y[0]+y[2]<0:
        path_dir_anger[x]="down"
      else:
        path_dir_anger[x]="right"
  else:
    if y[0]>0 and y[2]>0:
      if y[1]<0:
        path_dir_anger[x]="forward"
        if y[0]>15:
          path_dir_anger[x]="down"
      else:
        path_dir_anger[x]="forward"
      
    else:
      if y[0]<0 and y[1]<0 and y[2]<0:
        if y[0]<-15:
          path_dir_anger[x]="down"
        else:
          if y[2]<-12 or y[1]<-10:
            path_dir_anger[x]="right"
          else:
            path_dir_anger[x]="forward"
      else:
        if y[0]<-13:
          path_dir_anger[x]="down"
        else:
          if y[0]+y[1]<-10:
            path_dir_anger[x]="forward"
          else:
            if y[1]>3:
              path_dir_anger[x]="forward"
            else:
              if y[0]>10:
                path_dir_anger[x]="up"
              else:
                if y[0]<-4 and y[1]>0 and y[2]>0:
                  path_dir_anger[x]="down"
                else:
                  if y[0]>y[1] and y[0]>y[2] and y[0]>0 and y[1]>0:
                    path_dir_anger[x]="up"
                  else:
                    if y[0]>0 and y[1]+y[2]<-11:
                      path_dir_anger[x]="forward"
                    else:
                      if y[0]<-1 and y[1]+y[2]>4:
                        path_dir_anger[x]="forward"
                      else:
                        path_dir_anger[x]="forward"


# In[127]:


path_dir_contempt={}
for x, y in new_contempt.items():
  #print("pitch:",y[0],"yaw:",y[1],"roll:",y[2])
  if y[1]<y[0] and y[1]<y[2] and y[1]<0 and y[1]<-15:
    if y[0]>0 and y[2]<0:
      if y[0]<3:
        path_dir_contempt[x]="left"
      else:
        path_dir_contempt[x]="gaze-up"
    elif y[0]<0 and y[2]<0:
      if y[0]+y[2]>-15:
        path_dir_contempt[x]="left"
      else:
        path_dir_contempt[x]="gaze-down"
    else:
      path_dir_contempt[x]="left"
  elif y[1]>y[0] and y[1]>y[2] and y[1]>0 and y[1]>5:
    if y[0]+y[2]>-9 and y[0]<0 and y[2]<0:
      path_dir_contempt[x]="right"
    else:
      if y[0]+y[2]>0:
        path_dir_contempt[x]="gaze-up"
      elif y[0]+y[2]<0:
        path_dir_contempt[x]="gaze-down"
      else:
        path_dir_contempt[x]="right"
  else:
    if y[0]>0 and y[2]>0:
      if y[1]<0:
        path_dir_contempt[x]="forward"
        if y[0]>15:
          path_dir_contempt[x]="gaze-down"
      else:
        path_dir_contempt[x]="forward"
      
    else:
      if y[0]<0 and y[1]<0 and y[2]<0:
        if y[0]<-15:
          path_dir_contempt[x]="gaze-down"
        else:
          if y[2]<-12 or y[1]<-10:
            path_dir_contempt[x]="right"
          else:
            path_dir_contempt[x]="forward"
      else:
        if y[0]<-13:
          path_dir_contempt[x]="gaze-down"
        else:
          if y[0]+y[1]<-10:
            path_dir_contempt[x]="forward"
          else:
            if y[1]>3:
              path_dir_contempt[x]="forward"
            else:
              if y[0]>10:
                path_dir_contempt[x]="gaze-up"
              else:
                if y[0]<-4 and y[1]>0 and y[2]>0:
                  path_dir_contempt[x]="gaze-down"
                else:
                  if y[0]>y[1] and y[0]>y[2] and y[0]>0 and y[1]>0:
                    path_dir_contempt[x]="gaze-up"
                  else:
                    if y[0]>0 and y[1]+y[2]<-11:
                      path_dir_contempt[x]="forward"
                    else:
                      if y[0]<-1 and y[1]+y[2]>4:
                        path_dir_contempt[x]="forward"
                      else:
                        path_dir_contempt[x]="forward"


# In[129]:


path_dir_disgust={}
for x, y in new_disgust.items():
  #print("pitch:",y[0],"yaw:",y[1],"roll:",y[2])
  if y[1]<y[0] and y[1]<y[2] and y[1]<0 and y[1]<-15:
    if y[0]>0 and y[2]<0:
      if y[0]<3:
        path_dir_disgust[x]="left"
      else:
        path_dir_disgust[x]="up"
    elif y[0]<0 and y[2]<0:
      if y[0]+y[2]>-15:
        path_dir_disgust[x]="left"
      else:
        path_dir_disgust[x]="down"
    else:
      path_dir_disgust[x]="left"
  elif y[1]>y[0] and y[1]>y[2] and y[1]>0 and y[1]>5:
    if y[0]+y[2]>-9 and y[0]<0 and y[2]<0:
      path_dir_disgust[x]="right"
    else:
      if y[0]+y[2]>0:
        path_dir_disgust[x]="up"
      elif y[0]+y[2]<0:
        path_dir_disgust[x]="down"
      else:
        path_dir_disgust[x]="right"
  else:
    if y[0]>0 and y[2]>0:
      if y[1]<0:
        path_dir_disgust[x]="forward"
        if y[0]>15:
          path_dir_disgust[x]="down"
      else:
        path_dir_disgust[x]="forward"
      
    else:
      if y[0]<0 and y[1]<0 and y[2]<0:
        if y[0]<-15:
          path_dir_disgust[x]="down"
        else:
          if y[2]<-12 or y[1]<-10:
            path_dir_disgust[x]="right"
          else:
            path_dir_disgust[x]="forward"
      else:
        if y[0]<-13:
          path_dir_disgust[x]="down"
        else:
          if y[0]+y[1]<-10:
            path_dir_disgust[x]="forward"
          else:
            if y[1]>3:
              path_dir_disgust[x]="forward"
            else:
              if y[0]>10:
                path_dir_disgust[x]="up"
              else:
                if y[0]<-4 and y[1]>0 and y[2]>0:
                  path_dir_disgust[x]="down"
                else:
                  if y[0]>y[1] and y[0]>y[2] and y[0]>0 and y[1]>0:
                    path_dir_disgust[x]="up"
                  else:
                    if y[0]>0 and y[1]+y[2]<-11:
                      path_dir_disgust[x]="forward"
                    else:
                      if y[0]<-1 and y[1]+y[2]>4:
                        path_dir_disgust[x]="forward"
                      else:
                        path_dir_disgust[x]="forward"


# In[131]:


path_dir_fear={}
for x, y in new_fear.items():
  #print("pitch:",y[0],"yaw:",y[1],"roll:",y[2])
  if y[1]<y[0] and y[1]<y[2] and y[1]<0 and y[1]<-15:
    if y[0]>0 and y[2]<0:
      if y[0]<3:
        path_dir_fear[x]="left"
      else:
        path_dir_fear[x]="up"
    elif y[0]<0 and y[2]<0:
      if y[0]+y[2]>-15:
        path_dir_fear[x]="left"
      else:
        path_dir_fear[x]="down"
    else:
      path_dir_fear[x]="left"
  elif y[1]>y[0] and y[1]>y[2] and y[1]>0 and y[1]>5:
    if y[0]+y[2]>-9 and y[0]<0 and y[2]<0:
      path_dir_fear[x]="right"
    else:
      if y[0]+y[2]>0:
        path_dir_fear[x]="up"
      elif y[0]+y[2]<0:
        path_dir_fear[x]="down"
      else:
        path_dir_fear[x]="right"
  else:
    if y[0]>0 and y[2]>0:
      if y[1]<0:
        path_dir_fear[x]="forward"
        if y[0]>15:
          path_dir_fear[x]="down"
      else:
        path_dir_fear[x]="forward"
      
    else:
      if y[0]<0 and y[1]<0 and y[2]<0:
        if y[0]<-15:
          path_dir_fear[x]="down"
        else:
          if y[2]<-12 or y[1]<-10:
            path_dir_fear[x]="right"
          else:
            path_dir_fear[x]="forward"
      else:
        if y[0]<-13:
          path_dir_fear[x]="down"
        else:
          if y[0]+y[1]<-10:
            path_dir_fear[x]="forward"
          else:
            if y[1]>3:
              path_dir_fear[x]="forward"
            else:
              if y[0]>10:
                path_dir_fear[x]="up"
              else:
                if y[0]<-4 and y[1]>0 and y[2]>0:
                  path_dir_fear[x]="down"
                else:
                  if y[0]>y[1] and y[0]>y[2] and y[0]>0 and y[1]>0:
                    path_dir_fear[x]="up"
                  else:
                    if y[0]>0 and y[1]+y[2]<-11:
                      path_dir_fear[x]="forward"
                    else:
                      if y[0]<-1 and y[1]+y[2]>4:
                        path_dir_fear[x]="forward"
                      else:
                        path_dir_fear[x]="forward"


# In[133]:


path_dir_happy={}
for x, y in new_happy.items():
  #print("pitch:",y[0],"yaw:",y[1],"roll:",y[2])
  if y[1]<y[0] and y[1]<y[2] and y[1]<0 and y[1]<-15:
    if y[0]>0 and y[2]<0:
      if y[0]<3:
        path_dir_happy[x]="left"
      else:
        path_dir_happy[x]="up"
    elif y[0]<0 and y[2]<0:
      if y[0]+y[2]>-15:
        path_dir_happy[x]="left"
      else:
        path_dir_happy[x]="down"
    else:
      path_dir_happy[x]="left"
  elif y[1]>y[0] and y[1]>y[2] and y[1]>0 and y[1]>5:
    if y[0]+y[2]>-9 and y[0]<0 and y[2]<0:
      path_dir_happy[x]="right"
    else:
      if y[0]+y[2]>0:
        path_dir_happy[x]="up"
      elif y[0]+y[2]<0:
        path_dir_happy[x]="down"
      else:
        path_dir_happy[x]="right"
  else:
    if y[0]>0 and y[2]>0:
      if y[1]<0:
        path_dir_happy[x]="forward"
        if y[0]>15:
          path_dir_happy[x]="down"
      else:
        path_dir_happy[x]="forward"
      
    else:
      if y[0]<0 and y[1]<0 and y[2]<0:
        if y[0]<-15:
          path_dir_happy[x]="down"
        else:
          if y[2]<-12 or y[1]<-10:
            path_dir_happy[x]="right"
          else:
            path_dir_happy[x]="forward"
      else:
        if y[0]<-13:
          path_dir_happy[x]="down"
        else:
          if y[0]+y[1]<-10:
            path_dir_happy[x]="forward"
          else:
            if y[1]>3:
              path_dir_happy[x]="forward"
            else:
              if y[0]>10:
                path_dir_happy[x]="up"
              else:
                if y[0]<-4 and y[1]>0 and y[2]>0:
                  path_dir_happy[x]="down"
                else:
                  if y[0]>y[1] and y[0]>y[2] and y[0]>0 and y[1]>0:
                    path_dir_happy[x]="up"
                  else:
                    if y[0]>0 and y[1]+y[2]<-11:
                      path_dir_happy[x]="forward"
                    else:
                      if y[0]<-1 and y[1]+y[2]>4:
                        path_dir_happy[x]="forward"
                      else:
                        path_dir_happy[x]="forward"


# In[135]:


path_dir_sad={}
for x, y in new_sad.items():
  #print("pitch:",y[0],"yaw:",y[1],"roll:",y[2])
  if y[1]<y[0] and y[1]<y[2] and y[1]<0 and y[1]<-15:
    if y[0]>0 and y[2]<0:
      if y[0]<3:
        path_dir_sad[x]="left"
      else:
        path_dir_sad[x]="up"
    elif y[0]<0 and y[2]<0:
      if y[0]+y[2]>-15:
        path_dir_sad[x]="left"
      else:
        path_dir_sad[x]="down"
    else:
      path_dir_sad[x]="left"
  elif y[1]>y[0] and y[1]>y[2] and y[1]>0 and y[1]>5:
    if y[0]+y[2]>-9 and y[0]<0 and y[2]<0:
      path_dir_sad[x]="right"
    else:
      if y[0]+y[2]>0:
        path_dir_sad[x]="up"
      elif y[0]+y[2]<0:
        path_dir_sad[x]="down"
      else:
        path_dir_sad[x]="right"
  else:
    if y[0]>0 and y[2]>0:
      if y[1]<0:
        path_dir_sad[x]="forward"
        if y[0]>15:
          path_dir_sad[x]="down"
      else:
        path_dir_sad[x]="forward"
      
    else:
      if y[0]<0 and y[1]<0 and y[2]<0:
        if y[0]<-15:
          path_dir_sad[x]="down"
        else:
          if y[2]<-12 or y[1]<-10:
            path_dir_sad[x]="right"
          else:
            path_dir_sad[x]="forward"
      else:
        if y[0]<-13:
          path_dir_sad[x]="down"
        else:
          if y[0]+y[1]<-10:
            path_dir_sad[x]="forward"
          else:
            if y[1]>3:
              path_dir_sad[x]="forward"
            else:
              if y[0]>10:
                path_dir_sad[x]="up"
              else:
                if y[0]<-4 and y[1]>0 and y[2]>0:
                  path_dir_sad[x]="down"
                else:
                  if y[0]>y[1] and y[0]>y[2] and y[0]>0 and y[1]>0:
                    path_dir_sad[x]="up"
                  else:
                    if y[0]>0 and y[1]+y[2]<-11:
                      path_dir_sad[x]="forward"
                    else:
                      if y[0]<-1 and y[1]+y[2]>4:
                        path_dir_sad[x]="forward"
                      else:
                        path_dir_sad[x]="forward"


# In[137]:


path_dir_surprise={}
for x, y in new_surprise.items():
  #print("pitch:",y[0],"yaw:",y[1],"roll:",y[2])
  if y[1]<y[0] and y[1]<y[2] and y[1]<0 and y[1]<-15:
    if y[0]>0 and y[2]<0:
      if y[0]<3:
        path_dir_surprise[x]="left"
      else:
        path_dir_surprise[x]="up"
    elif y[0]<0 and y[2]<0:
      if y[0]+y[2]>-15:
        path_dir_surprise[x]="left"
      else:
        path_dir_surprise[x]="down"
    else:
      path_dir_surprise[x]="left"
  elif y[1]>y[0] and y[1]>y[2] and y[1]>0 and y[1]>5:
    if y[0]+y[2]>-9 and y[0]<0 and y[2]<0:
      path_dir_surprise[x]="right"
    else:
      if y[0]+y[2]>0:
        path_dir_surprise[x]="up"
      elif y[0]+y[2]<0:
        path_dir_surprise[x]="down"
      else:
        path_dir_surprise[x]="right"
  else:
    if y[0]>0 and y[2]>0:
      if y[1]<0:
        path_dir_surprise[x]="forward"
        if y[0]>15:
          path_dir_surprise[x]="down"
      else:
        path_dir_surprise[x]="forward"
      
    else:
      if y[0]<0 and y[1]<0 and y[2]<0:
        if y[0]<-15:
          path_dir_surprise[x]="down"
        else:
          if y[2]<-12 or y[1]<-10:
            path_dir_surprise[x]="right"
          else:
            path_dir_surprise[x]="forward"
      else:
        if y[0]<-13:
          path_dir_surprise[x]="down"
        else:
          if y[0]+y[1]<-10:
            path_dir_surprise[x]="forward"
          else:
            if y[1]>3:
              path_dir_surprise[x]="forward"
            else:
              if y[0]>10:
                path_dir_surprise[x]="up"
              else:
                if y[0]<-4 and y[1]>0 and y[2]>0:
                  path_dir_surprise[x]="down"
                else:
                  if y[0]>y[1] and y[0]>y[2] and y[0]>0 and y[1]>0:
                    path_dir_surprise[x]="up"
                  else:
                    if y[0]>0 and y[1]+y[2]<-11:
                      path_dir_surprise[x]="forward"
                    else:
                      if y[0]<-1 and y[1]+y[2]>4:
                        path_dir_surprise[x]="forward"
                      else:
                        path_dir_surprise[x]="forward"


# In[139]:


path_dir_neutral={}
for x, y in new_neutral.items():
  #print("pitch:",y[0],"yaw:",y[1],"roll:",y[2])
  if y[1]<y[0] and y[1]<y[2] and y[1]<0 and y[1]<-15:
    if y[0]>0 and y[2]<0:
      if y[0]<3:
        path_dir_neutral[x]="left"
      else:
        path_dir_neutral[x]="up"
    elif y[0]<0 and y[2]<0:
      if y[0]+y[2]>-15:
        path_dir_neutral[x]="left"
      else:
        path_dir_neutral[x]="down"
    else:
      path_dir_neutral[x]="left"
  elif y[1]>y[0] and y[1]>y[2] and y[1]>0 and y[1]>5:
    if y[0]+y[2]>-9 and y[0]<0 and y[2]<0:
      path_dir_neutral[x]="right"
    else:
      if y[0]+y[2]>0:
        path_dir_neutral[x]="up"
      elif y[0]+y[2]<0:
        path_dir_neutral[x]="down"
      else:
        path_dir_neutral[x]="right"
  else:
    if y[0]>0 and y[2]>0:
      if y[1]<0:
        path_dir_neutral[x]="forward"
        if y[0]>15:
          path_dir_neutral[x]="down"
      else:
        path_dir_neutral[x]="forward"
      
    else:
      if y[0]<0 and y[1]<0 and y[2]<0:
        if y[0]<-15:
          path_dir_neutral[x]="down"
        else:
          if y[2]<-12 or y[1]<-10:
            path_dir_neutral[x]="right"
          else:
            path_dir_neutral[x]="forward"
      else:
        if y[0]<-13:
          path_dir_neutral[x]="down"
        else:
          if y[0]+y[1]<-10:
            path_dir_neutral[x]="forward"
          else:
            if y[1]>3:
              path_dir_neutral[x]="forward"
            else:
              if y[0]>10:
                path_dir_neutral[x]="up"
              else:
                if y[0]<-4 and y[1]>0 and y[2]>0:
                  path_dir_neutral[x]="down"
                else:
                  if y[0]>y[1] and y[0]>y[2] and y[0]>0 and y[1]>0:
                    path_dir_neutral[x]="up"
                  else:
                    if y[0]>0 and y[1]+y[2]<-11:
                      path_dir_neutral[x]="forward"
                    else:
                      if y[0]<-1 and y[1]+y[2]>4:
                        path_dir_neutral[x]="forward"
                      else:
                        path_dir_neutral[x]="forward"


# In[166]:


import os, os.path, shutil
#Put all images in one folder
anger_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Anger/"
anger_images = [f for f in os.listdir(anger_folder_path) if os.path.isfile(os.path.join(anger_folder_path, f))]

#Put all images in one folder
contempt_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Contempt/"
contempt_images = [f for f in os.listdir(contempt_folder_path) if os.path.isfile(os.path.join(contempt_folder_path, f))]

#Put all images in one folder
disgust_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Disgust/"
disgust_images = [f for f in os.listdir(disgust_folder_path) if os.path.isfile(os.path.join(disgust_folder_path, f))]

#Put all images in one folder
fear_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Fear/"
fear_images = [f for f in os.listdir(fear_folder_path) if os.path.isfile(os.path.join(fear_folder_path, f))]

#Put all images in one folder
happy_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Happy/"
happy_images = [f for f in os.listdir(happy_folder_path) if os.path.isfile(os.path.join(happy_folder_path, f))]

#Put all images in one folder
neutral_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Neutral/"
neutral_images = [f for f in os.listdir(neutral_folder_path) if os.path.isfile(os.path.join(neutral_folder_path, f))]

#Put all images in one folder
sad_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Sad/"
sad_images = [f for f in os.listdir(sad_folder_path) if os.path.isfile(os.path.join(sad_folder_path, f))]

#Put all images in one folder
surprise_folder_path="/mnt/nas4/Zhansaya_affectnet/train_set/balanced/train/Surprise/"
surprise_images = [f for f in os.listdir(surprise_folder_path) if os.path.isfile(os.path.join(surprise_folder_path, f))]


# In[165]:


for image in anger_images:
    for x,y in path_dir_anger.items():
        if x == str(anger_folder_path+image):
            if y =='forward':
                f_name='Forward/'
                new_path2 = anger_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='left':
                f_name='Left/'
                new_path2 = anger_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='right':
                f_name='Right/'
                new_path2 = anger_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='up':
                f_name='Up/'
                new_path2 = anger_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='down':
                f_name='Down/'
                new_path2 = anger_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)


# In[167]:


for image in contempt_images:
    for x,y in path_dir_contempt.items():
        if x == str(contempt_folder_path+image):
            if y =='forward':
                f_name='Forward/'
                new_path2 = contempt_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='left':
                f_name='Left/'
                new_path2 = contempt_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='right':
                f_name='Right/'
                new_path2 = contempt_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='up':
                f_name='Up/'
                new_path2 = contempt_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='down':
                f_name='Down/'
                new_path2 = contempt_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)


# In[168]:


for image in disgust_images:
    for x,y in path_dir_disgust.items():
        if x == str(disgust_folder_path+image):
            if y =='forward':
                f_name='Forward/'
                new_path2 = disgust_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='left':
                f_name='Left/'
                new_path2 = disgust_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='right':
                f_name='Right/'
                new_path2 = disgust_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='up':
                f_name='Up/'
                new_path2 = disgust_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='down':
                f_name='Down/'
                new_path2 = disgust_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)


# In[169]:


for image in fear_images:
    for x,y in path_dir_fear.items():
        if x == str(fear_folder_path+image):
            if y =='forward':
                f_name='Forward/'
                new_path2 = fear_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='left':
                f_name='Left/'
                new_path2 = fear_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='right':
                f_name='Right/'
                new_path2 = fear_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='up':
                f_name='Up/'
                new_path2 = fear_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='down':
                f_name='Down/'
                new_path2 = fear_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)


# In[170]:


for image in happy_images:
    for x,y in path_dir_happy.items():
        if x == str(happy_folder_path+image):
            if y =='forward':
                f_name='Forward/'
                new_path2 = happy_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='left':
                f_name='Left/'
                new_path2 = happy_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='right':
                f_name='Right/'
                new_path2 = happy_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='up':
                f_name='Up/'
                new_path2 = happy_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='down':
                f_name='Down/'
                new_path2 = happy_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)


# In[171]:


for image in neutral_images:
    for x,y in path_dir_neutral.items():
        if x == str(neutral_folder_path+image):
            if y =='forward':
                f_name='Forward/'
                new_path2 = neutral_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='left':
                f_name='Left/'
                new_path2 = neutral_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='right':
                f_name='Right/'
                new_path2 = neutral_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='up':
                f_name='Up/'
                new_path2 = neutral_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='down':
                f_name='Down/'
                new_path2 = neutral_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)


# In[172]:


for image in sad_images:
    for x,y in path_dir_sad.items():
        if x == str(sad_folder_path+image):
            if y =='forward':
                f_name='Forward/'
                new_path2 = sad_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='left':
                f_name='Left/'
                new_path2 = sad_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='right':
                f_name='Right/'
                new_path2 = sad_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='up':
                f_name='Up/'
                new_path2 = sad_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='down':
                f_name='Down/'
                new_path2 = sad_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)


# In[173]:


for image in surprise_images:
    for x,y in path_dir_surprise.items():
        if x == str(surprise_folder_path+image):
            if y =='forward':
                f_name='Forward/'
                new_path2 = surprise_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='left':
                f_name='Left/'
                new_path2 = surprise_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='right':
                f_name='Right/'
                new_path2 = surprise_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='up':
                f_name='Up/'
                new_path2 = surprise_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)
            elif y =='down':
                f_name='Down/'
                new_path2 = surprise_folder_path+f_name
                if not os.path.exists(new_path2):
                    os.makedirs(new_path2)
                new_image_path = os.path.join(new_path2, image)
                shutil.move(x, new_image_path)


# In[294]:


folder_path = "/mnt/nas4/Zhansaya_affectnet/train_set/balanced_directions/train/Surprise/Forward/"

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

new_path= '/mnt/nas4/Zhansaya_affectnet/train_set/balanced_directions/test/Surprise/Forward/'

for i in range(20):
    old_image_path = os.path.join(folder_path, images[i])
    new_image_path = os.path.join(new_path, images[i])
    shutil.move(old_image_path, new_image_path)


# In[295]:


folder_path = "/mnt/nas4/Zhansaya_affectnet/train_set/balanced_directions/train/Surprise/Left/"

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

new_path= '/mnt/nas4/Zhansaya_affectnet/train_set/balanced_directions/test/Surprise/Left/'

for i in range(20):
    old_image_path = os.path.join(folder_path, images[i])
    new_image_path = os.path.join(new_path, images[i])
    shutil.move(old_image_path, new_image_path)


# In[296]:


folder_path = "/mnt/nas4/Zhansaya_affectnet/train_set/balanced_directions/train/Surprise/Right/"

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

new_path= '/mnt/nas4/Zhansaya_affectnet/train_set/balanced_directions/test/Surprise/Right/'

for i in range(20):
    old_image_path = os.path.join(folder_path, images[i])
    new_image_path = os.path.join(new_path, images[i])
    shutil.move(old_image_path, new_image_path)


# In[297]:


folder_path = "/mnt/nas4/Zhansaya_affectnet/train_set/balanced_directions/train/Surprise/Up/"

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

new_path= '/mnt/nas4/Zhansaya_affectnet/train_set/balanced_directions/test/Surprise/Up/'

for i in range(20):
    old_image_path = os.path.join(folder_path, images[i])
    new_image_path = os.path.join(new_path, images[i])
    shutil.move(old_image_path, new_image_path)


# In[298]:


folder_path = "/mnt/nas4/Zhansaya_affectnet/train_set/balanced_directions/train/Surprise/Down/"

images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

new_path= '/mnt/nas4/Zhansaya_affectnet/train_set/balanced_directions/test/Surprise/Down/'

for i in range(20):
    old_image_path = os.path.join(folder_path, images[i])
    new_image_path = os.path.join(new_path, images[i])
    shutil.move(old_image_path, new_image_path)


# In[ ]:




