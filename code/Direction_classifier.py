#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv ('/content/drive/MyDrive/Emotion Recognition/int_train_directions.csv')
df=df.drop(['Direction'], axis=1)


# In[1]:


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


# In[ ]:


print(sns.countplot(x = 'Int_direction', data=df))


# In[ ]:


df=df.drop(['Unnamed: 0'], axis=1)


# In[ ]:


# create a boolean mask to identify the first occurrence of each value
mask_0 = (df['Int_direction'] == 0) & (df['Int_direction'].shift(1) != 0)
mask_1 = (df['Int_direction'] == 1) & (df['Int_direction'].shift(1) != 1)
mask_2 = (df['Int_direction'] == 2) & (df['Int_direction'].shift(1) != 2)
mask_3 = (df['Int_direction'] == 3) & (df['Int_direction'].shift(1) != 3)
mask_4 = (df['Int_direction'] == 4) & (df['Int_direction'].shift(1) != 4)

# use the boolean mask and idxmin method to get the row number of the first occurrence of each value
idx_0 = np.where(mask_0)[0][0]
idx_1 = np.where(mask_1)[0][0]
idx_2 = np.where(mask_2)[0][0]
idx_3 = np.where(mask_3)[0][0]
idx_4 = np.where(mask_4)[0][0]


print("Row number of first 0:", idx_0)
print("Row number of first 1:", idx_1)
print("Row number of first 2:", idx_2)
print("Row number of first 3:", idx_3)
print("Row number of first 4:", idx_4)


# In[ ]:


# select rows 0-2 and 4-5 and create a new DataFrame
df_range0 = df.iloc[0:426]  # select rows 0-2
df_range1 = df.iloc[539:965]  # select rows 4-5
df_range2 = df.iloc[1088:1514]  # select rows 0-2
df_range3 = df.iloc[1649:2074]  # select rows 4-5
df_range4 = df.iloc[2076:2502]  # select rows 0-2
df_new = pd.concat([df_range0, df_range1,df_range2, df_range3,df_range4], ignore_index=True)


# In[ ]:


X = df_new.iloc[:, 1:-1]
y = df_new.iloc[:, -1]


# In[ ]:


# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)


# In[ ]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)


# In[ ]:


def get_class_distribution(obj):
    count_dict = {
        "forward": 0,
        "left": 0,
        "right": 0,
        "up": 0,
        "down": 0
    }
    
    for i in obj:
        if i == 0: 
            count_dict['forward'] += 1
        elif i == 1: 
            count_dict['left'] += 1
        elif i == 2: 
            count_dict['right'] += 1
        elif i == 3: 
            count_dict['up'] += 1
        elif i == 4: 
            count_dict['down'] += 1             
        else:
            print("Check classes.")
            
    return count_dict


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25,7))
# Train
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_train)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[0]).set_title('Class Distribution in Train Set')
# Validation
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_val)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[1]).set_title('Class Distribution in Val Set')
# Test
sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(y_test)]).melt(), x = "variable", y="value", hue="variable",  ax=axes[2]).set_title('Class Distribution in Test Set')


# In[ ]:


class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())


# In[ ]:


target_list = []
for _, t in train_dataset:
    target_list.append(t)
    
target_list = torch.tensor(target_list)


# In[ ]:


class_count = [i for i in get_class_distribution(y_train).values()]
class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
print(class_weights)


# In[ ]:


class_weights_all = class_weights[target_list]


# In[ ]:


weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)


# In[ ]:


EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.0007
NUM_FEATURES = len(X.columns)
NUM_CLASSES = 5


# In[ ]:


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          sampler=weighted_sampler
)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


# In[ ]:


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


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)


# In[ ]:


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc


# In[ ]:


accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}


# In[ ]:


print("Begin training.")
for e in tqdm(range(1, EPOCHS+1)):
    
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = model(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
                              
    
    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')


# In[ ]:


# Create dataframes
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')


# In[ ]:


y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        _, y_pred_tags = torch.max(y_test_pred, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]


# In[ ]:


print(classification_report(y_test, y_pred_list))


# In[ ]:


# save the trained model to a file
torch.save(model.state_dict(), '/content/drive/MyDrive/Emotion Recognition/nnmodel.pth')


# In[ ]:


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
data = pd.read_csv('/content/drive/MyDrive/Emotion Recognition/test_directions.csv')

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
state_dict = torch.load('/content/drive/MyDrive/Emotion Recognition/nnmodel.pth')

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


# In[ ]:


num=0
for i in predicted_directions:
  if i=='0':
    num=num+1

print(num)


# In[ ]:


num=0
for i in predicted_directions:
  if i=='1':
    num=num+1

print(num)


# In[ ]:


num=0
for i in predicted_directions:
  if i=='2':
    num=num+1

print(num)


# In[ ]:


num=0
for i in predicted_directions:
  if i=='3':
    num=num+1

print(num)


# In[ ]:


num=0
for i in predicted_directions:
  if i=='4':
    num=num+1

print(num)


# In[ ]:


# Create a new column with the predicted directions
data['Direction'] = predicted_directions

# Save the DataFrame to a CSV file
data.to_csv('/content/drive/MyDrive/Emotion Recognition/test_directions_predicted.csv', index=False)

