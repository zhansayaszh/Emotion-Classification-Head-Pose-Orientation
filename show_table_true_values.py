#!/usr/bin/env python
# coding: utf-8

# In[8]:


import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define the transforms to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset from the folder structure and apply the transforms
dataset = ImageFolder('/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/TEST/', transform=transform)

# Create a data loader to load the images in batches
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load the CSV file containing the image filenames and their true labels
df = pd.read_csv('/home/snake/Desktop/imbalanced/new_TEST.csv')

model = models.mobilenet_v3_small(pretrained=False, num_classes=8) # we do not specify pretrained=True, i.e. do not load default weights
# Load pre-trained state dictionary
checkpoint = torch.load('/home/snake/Desktop/imbalanced/finalmobilenet2.pth')
# Modify classifier to match checkpoint
model.classifier[3] = torch.nn.Linear(1024, 8)
# Load state dictionary into model
model.load_state_dict(checkpoint)

# Set the model to evaluate mode
model.eval()

# Initialize lists to store the true positive filenames and their predicted labels
true_positives = []
predicted_labels = []

# Iterate through the batches of images
for batch, (images, labels) in enumerate(dataloader):
    # Forward pass the images through the model to get predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Convert the predicted labels to a list
    preds = preds.tolist()

    # Iterate through the batch and compare the predicted labels to the true labels
    for i in range(len(labels)):
        # Check if the prediction is correct
        if preds[i] == labels[i]:
            # Get the filename of the image
            filename = dataset.samples[batch*32+i][0]
            # Append the filename and predicted label to their respective lists
            true_positives.append(filename)
            predicted_labels.append(preds[i])


# In[9]:



# Assuming you have a PyTorch model and dataloader set up
model.eval()
y_true = []
y_pred = []
for images, labels in dataloader:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    y_true += labels.tolist()
    y_pred += predicted.tolist()

# Generate confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)

emotion_true_values={}
for k in range(8):
    # Identify true positive indices for a given class
    class_index = k  # Replace with the index of the class you're interested in
    true_positives = np.diag(conf_mat)[class_index]

    # Get filenames for true positive predictions
    all_filenames = dataloader.dataset.imgs
    true_positive_filenames = [all_filenames[i] for i in range(len(y_true)) if y_true[i] == class_index and y_pred[i] == class_index]
    emotion_true_values[k] = true_positive_filenames


# In[12]:


class_names=['Anger','Contempt','Disgust','Fear','Happy','Neutral','Sad','Surprise']
# Create a heatmap using seaborn
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g',xticklabels=class_names,yticklabels=class_names,)

# Add labels and title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Proposed MobileNet Model Confusion Matrix')

# Display the heatmap
plt.show()


# In[13]:


emotionsdf=pd.DataFrame({'Anger': pd.Series(emotion_true_values[0]), 'Contempt': pd.Series(emotion_true_values[1]),
                       'Disgust': pd.Series(emotion_true_values[2]), 'Fear': pd.Series(emotion_true_values[3]),
                      'Happy': pd.Series(emotion_true_values[4]), 'Neutral': pd.Series(emotion_true_values[5]),
                      'Sad': pd.Series(emotion_true_values[6]), 'Surprise': pd.Series(emotion_true_values[7])})


# In[14]:


emotionsdf.to_csv('/home/snake/Desktop/imbalanced/emotions_true_values2.csv')


# In[15]:


import numpy as np

anger=[0,0,0,0,0]
for k in range(len(emotion_true_values[0])):
    if 'Forward'== emotion_true_values[0][k][0].split('/')[-2]:
        anger[0]+=1
    elif 'Left'== emotion_true_values[0][k][0].split('/')[-2]:
        anger[1]+=1
    elif 'Right'== emotion_true_values[0][k][0].split('/')[-2]:
        anger[2]+=1
    elif 'Up'== emotion_true_values[0][k][0].split('/')[-2]:
        anger[3]+=1
    elif 'Down'== emotion_true_values[0][k][0].split('/')[-2]:
        anger[4]+=1

print("Anger:",len(emotion_true_values[0]),"\n",
      "Forward:",anger[0],'\n',
     "Left:",anger[1],'\n',
     "Right:",anger[2],'\n',
     "Up:",anger[3],'\n',
     "Down:",anger[4],'\n')

print("Mean:",np.average(anger))
print("Variance:",np.var(anger))
print("Standard deviation:",np.std(anger),'\n')

contempt=[0,0,0,0,0]
for k in range(len(emotion_true_values[1])):
    if 'Forward'== emotion_true_values[1][k][0].split('/')[-2]:
        contempt[0]+=1
    elif 'Left'== emotion_true_values[1][k][0].split('/')[-2]:
        contempt[1]+=1
    elif 'Right'== emotion_true_values[1][k][0].split('/')[-2]:
        contempt[2]+=1
    elif 'Up'== emotion_true_values[1][k][0].split('/')[-2]:
        contempt[3]+=1
    elif 'Down'== emotion_true_values[1][k][0].split('/')[-2]:
        contempt[4]+=1

print('Contempt:',len(emotion_true_values[1]),"\n",
    "Forward:",contempt[0],'\n',
     "Left:",contempt[1],'\n',
     "Right:",contempt[2],'\n',
     "Up:",contempt[3],'\n',
     "Down:",contempt[4],'\n')

print("Mean:",np.average(contempt))
print("Variance:",np.var(contempt))
print("Standard deviation:",np.std(contempt),'\n')

disgust=[0,0,0,0,0]
for k in range(len(emotion_true_values[2])):
    if 'Forward'== emotion_true_values[2][k][0].split('/')[-2]:
        disgust[0]+=1
    elif 'Left'== emotion_true_values[2][k][0].split('/')[-2]:
        disgust[1]+=1
    elif 'Right'== emotion_true_values[2][k][0].split('/')[-2]:
        disgust[2]+=1
    elif 'Up'== emotion_true_values[2][k][0].split('/')[-2]:
        disgust[3]+=1
    elif 'Down'== emotion_true_values[2][k][0].split('/')[-2]:
        disgust[4]+=1

print('Disgust:',len(emotion_true_values[2]),"\n",
    "Forward:",disgust[0],'\n',
     "Left:",disgust[1],'\n',
     "Right:",disgust[2],'\n',
     "Up:",disgust[3],'\n',
     "Down:",disgust[4],'\n')

print("Mean:",np.average(disgust))
print("Variance:",np.var(disgust))
print("Standard deviation:",np.std(disgust),'\n')

fear=[0,0,0,0,0]
for k in range(len(emotion_true_values[3])):
    if 'Forward'== emotion_true_values[3][k][0].split('/')[-2]:
        fear[0]+=1
    elif 'Left'== emotion_true_values[3][k][0].split('/')[-2]:
        fear[1]+=1
    elif 'Right'== emotion_true_values[3][k][0].split('/')[-2]:
        fear[2]+=1
    elif 'Up'== emotion_true_values[3][k][0].split('/')[-2]:
        fear[3]+=1
    elif 'Down'== emotion_true_values[3][k][0].split('/')[-2]:
        fear[4]+=1

print('Fear:',len(emotion_true_values[3]),"\n",
    "Forward:",fear[0],'\n',
     "Left:",fear[1],'\n',
     "Right:",fear[2],'\n',
     "Up:",fear[3],'\n',
     "Down:",fear[4],'\n')

print("Mean:",np.average(fear))
print("Variance:",np.var(fear))
print("Standard deviation:",np.std(fear),'\n')

happy=[0,0,0,0,0]
for k in range(len(emotion_true_values[4])):
    if 'Forward'== emotion_true_values[4][k][0].split('/')[-2]:
        happy[0]+=1
    elif 'Left'== emotion_true_values[4][k][0].split('/')[-2]:
        happy[1]+=1
    elif 'Right'== emotion_true_values[4][k][0].split('/')[-2]:
        happy[2]+=1
    elif 'Up'== emotion_true_values[4][k][0].split('/')[-2]:
        happy[3]+=1
    elif 'Down'== emotion_true_values[4][k][0].split('/')[-2]:
        happy[4]+=1

print('Happy:',len(emotion_true_values[4]),"\n",
    "Forward:",happy[0],'\n',
     "Left:",happy[1],'\n',
     "Right:",happy[2],'\n',
     "Up:",happy[3],'\n',
     "Down:",happy[4],'\n')

print("Mean:",np.average(happy))
print("Variance:",np.var(happy))
print("Standard deviation:",np.std(happy),'\n')

neutral=[0,0,0,0,0]
for k in range(len(emotion_true_values[5])):
    if 'Forward'== emotion_true_values[5][k][0].split('/')[-2]:
        neutral[0]+=1
    elif 'Left'== emotion_true_values[5][k][0].split('/')[-2]:
        neutral[1]+=1
    elif 'Right'== emotion_true_values[5][k][0].split('/')[-2]:
        neutral[2]+=1
    elif 'Up'== emotion_true_values[5][k][0].split('/')[-2]:
        neutral[3]+=1
    elif 'Down'== emotion_true_values[5][k][0].split('/')[-2]:
        neutral[4]+=1

print('Neutral:',len(emotion_true_values[5]),"\n",
    "Forward:",neutral[0],'\n',
     "Left:",neutral[1],'\n',
     "Right:",neutral[2],'\n',
     "Up:",neutral[3],'\n',
     "Down:",neutral[4],'\n')

print("Mean:",np.average(neutral))
print("Variance:",np.var(neutral))
print("Standard deviation:",np.std(neutral),'\n')


sad=[0,0,0,0,0]
for k in range(len(emotion_true_values[6])):
    if 'Forward'== emotion_true_values[6][k][0].split('/')[-2]:
        sad[0]+=1
    elif 'Left'== emotion_true_values[6][k][0].split('/')[-2]:
        sad[1]+=1
    elif 'Right'== emotion_true_values[6][k][0].split('/')[-2]:
        sad[2]+=1
    elif 'Up'== emotion_true_values[6][k][0].split('/')[-2]:
        sad[3]+=1
    elif 'Down'== emotion_true_values[6][k][0].split('/')[-2]:
        sad[4]+=1

print('Sad:',len(emotion_true_values[6]),"\n",
    "Forward:",sad[0],'\n',
     "Left:",sad[1],'\n',
     "Right:",sad[2],'\n',
     "Up:",sad[3],'\n',
     "Down:",sad[4],'\n')

print("Mean:",np.average(sad))
print("Variance:",np.var(sad))
print("Standard deviation:",np.std(sad),'\n')


surprise=[0,0,0,0,0]
for k in range(len(emotion_true_values[7])):
    if 'Forward'== emotion_true_values[7][k][0].split('/')[-2]:
        surprise[0]+=1
    elif 'Left'== emotion_true_values[7][k][0].split('/')[-2]:
        surprise[1]+=1
    elif 'Right'== emotion_true_values[7][k][0].split('/')[-2]:
        surprise[2]+=1
    elif 'Up'== emotion_true_values[7][k][0].split('/')[-2]:
        surprise[3]+=1
    elif 'Down'== emotion_true_values[7][k][0].split('/')[-2]:
        surprise[4]+=1

print('Surprise:',len(emotion_true_values[7]),"\n",
    "Forward:",surprise[0],'\n',
     "Left:",surprise[1],'\n',
     "Right:",surprise[2],'\n',
     "Up:",surprise[3],'\n',
     "Down:",surprise[4],'\n')


print("Mean:",np.average(surprise))
print("Variance:",np.var(surprise))
print("Standard deviation:",np.std(surprise),'\n')


# In[2]:


#Calculate mean, variance, st deviation for each direction

forward=[14,19,12,13,24,11,14,20]
print("Forward:")
print("Mean:",np.average(forward))
print("Variance:",np.var(forward))
print("Standard deviation:",np.std(forward),'\n')

left=[12,15,14,17,16,18,17,18]

print("Left:")
print("Mean:",np.average(left))
print("Variance:",np.var(left))
print("Standard deviation:",np.std(left),'\n')

right=[8,16,16,18,24,16,14,15]
print("Right:")
print("Mean:",np.average(right))
print("Variance:",np.var(right))
print("Standard deviation:",np.std(right),'\n')

up=[11,15,14,14,25,14,12,14]
print("Up:")
print("Mean:",np.average(up))
print("Variance:",np.var(up))
print("Standard deviation:",np.std(up),'\n')

down=[9,13,12,15,24,16,18,17]
print("Down:")
print("Mean:",np.average(down))
print("Variance:",np.var(down))
print("Standard deviation:",np.std(down),'\n')

