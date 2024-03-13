#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import seaborn as sn
import pandas as pd
import torchnet.meter.confusionmeter as cm
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.ensemble import AdaBoostClassifier



torch.cuda.is_available()

torch.cuda.device_count()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Data augmentation and normalization for training
# Just normalization for validation & test
data_transforms = {
    'TRAIN': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'VAL': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'TEST': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = '/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['TRAIN', 'VAL', 'TEST']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['TRAIN', 'VAL', 'TEST']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['TRAIN', 'VAL', 'TEST']}
class_names = image_datasets['TRAIN'].classes

#lists for graph generation
epoch_counter_train = []
epoch_counter_val = []
train_loss = []
val_loss = []
train_acc = []
val_acc = []

#Train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch +1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['TRAIN', 'VAL']:
            if phase == 'TRAIN':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'TRAIN'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'TRAIN':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            #For graph generation
            if phase == "TRAIN":
                train_loss.append(running_loss/dataset_sizes[phase])
                train_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_train.append(epoch)
            if phase == "VAL":
                val_loss.append(running_loss/ dataset_sizes[phase])
                val_acc.append(running_corrects.double() / dataset_sizes[phase])
                epoch_counter_val.append(epoch)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            #for printing        
            if phase == "TRAIN":    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == "VAL":    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the best model
            if phase == 'VAL' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#MobileNet V3 small
mobmodel = torchvision.models.mobilenet_v3_small(pretrained=True)
num_ftrs = mobmodel.classifier[3].in_features
mobmodel.classifier[3] = nn.Linear(num_ftrs, 8)
mobmodel= mobmodel.to(device)
criterion = nn.CrossEntropyLoss()

# Using Adam as the parameter optimizer
optimizer_ft = optim.Adam(mobmodel.parameters(), lr = 0.001, betas=(0.9, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

mobmodel = train_model(mobmodel, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10) 

mobilenet_path='/home/snake/Desktop/imbalanced/finalmobilenet2.pth'
torch.save(mobmodel.state_dict(), mobilenet_path)
mobmodel.load_state_dict(torch.load(mobilenet_path))

#Test the accuracy with test data
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = mobmodel(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

from sklearn.metrics import f1_score
# Iterate over data.
y_true, y_pred = [], []
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        #outputs = model(inputs)
        predicted_outputs = mobmodel(inputs)
        _, predicted = torch.max(predicted_outputs, 1)
        total += labels.size(0)
        #print(total)
        correct += (predicted == labels).sum().item()
        #print(correct)
        #f1 score
        temp_true=labels.cpu().data.numpy()
        temp_pred=predicted.cpu().data.numpy()
        y_true+=temp_true.tolist()
        y_pred+=temp_pred.tolist()
        # Iterate through the batch and compare the predicted labels to the true labels
        
print('F1 Score:')
f1=f1_score(y_true,y_pred, average = 'macro')
print(f1)

t=[]
#Class wise testing accuracy
class_correct = list(0. for i in range(8))
class_total = list(0. for i in range(8))
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = mobmodel(inputs)
            _, predicted = torch.max(outputs, 1)
            point = (predicted == labels).squeeze()
            for j in range(len(labels)):
                label = labels[j]
                class_correct[label] += point[j].item()
                class_total[label] += 1
                if predicted[j] == labels[j]:
                    # Get the filename of the image
                    filename = image_datasets['TEST'].samples[i*32+j][0]
                        # Append the filename and predicted label to their respective lists
                    t.append(filename)


for i in range(8):
    print('Accuracy of %5s : %2d %%' % (
        class_names[i], 100 * class_correct[i] / class_total[i]))


#Get the confusion matrix for testing data
confusion_matrix = cm.ConfusionMeter(8)
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = mobmodel(inputs)
        _, predicted = torch.max(outputs, 1)
        confusion_matrix.add(predicted, labels)
    print(confusion_matrix.conf)
   
#Confusion matrix as a heatmap
con_m = confusion_matrix.conf
df_con_m = pd.DataFrame(con_m, index= [i for i in class_names], columns = [i for i in class_names])
sn.set(font_scale= 1.1)
sn.heatmap(df_con_m, annot=True,fmt='g' ,  annot_kws={"size" : 10}, cbar = False, cmap="Blues") 


import matplotlib.pyplot as plt

#Insert values of train loss and acc, and val loss and acc for each epoch
train_loss = [1.8509, 1.6720, 1.5844, 1.5562, 1.5124, 1.4631, 1.3533, 1.3150, 1.2778, 1.2799]
val_loss = [1.7016, 1.5622, 1.5270, 1.5581, 1.5457, 1.4963, 1.3579, 1.3640, 1.3598, 1.3478]
train_acc = [0.2877, 0.3645, 0.4079, 0.4077, 0.4274, 0.4483, 0.4922, 0.5051, 0.5214, 0.5210]
val_acc = [0.3517, 0.4000, 0.4417, 0.4175, 0.4267, 0.4567, 0.4983, 0.4942, 0.5008, 0.4983]

epochs = range(1, 11)

# plot the loss
plt.plot(epochs, train_loss, label='train_loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot the accuracy
plt.plot(epochs, train_acc, label='train_acc')
plt.plot(epochs, val_acc, label='val_acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Googlenet
gmodel = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
num_ftrs = gmodel.fc.in_features
gmodel.fc = nn.Linear(num_ftrs, 8)
gmodel= gmodel.to(device)
criterion = nn.CrossEntropyLoss()

# Using Adam as the parameter optimizer
optimizer_ft = optim.Adam(gmodel.parameters(), lr = 0.001, betas=(0.9, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

gmodel = train_model(gmodel, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)  


#Test the accuracy with test data
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = gmodel(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))


from sklearn.metrics import f1_score
# Iterate over data.
y_true, y_pred = [], []
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        #outputs = model(inputs)
        predicted_outputs = gmodel(inputs)
        _, predicted = torch.max(predicted_outputs, 1)
        total += labels.size(0)
        #print(total)
        correct += (predicted == labels).sum().item()
        #print(correct)
        #f1 score
        temp_true=labels.cpu().data.numpy()
        temp_pred=predicted.cpu().data.numpy()
        y_true+=temp_true.tolist()
        y_pred+=temp_pred.tolist()
        
print('F1 Score:')
f1=f1_score(y_true,y_pred, average = 'macro')
print(f1)


googlenet_path='/home/snake/Desktop/imbalanced/googlenet.pth'
torch.save(gmodel.state_dict(), googlenet_path)
gmodel.load_state_dict(torch.load(googlenet_path))

#Resnet18
resmodel = models.resnet18(pretrained=True)
num_ftrs = resmodel.fc.in_features
resmodel.fc = nn.Linear(num_ftrs, 8)
resmodel = resmodel.to(device)
criterion = nn.CrossEntropyLoss()

# Using Adam as the parameter optimizer
optimizer_ft = optim.Adam(resmodel.parameters(), lr = 0.001, betas=(0.9, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

resmodel = train_model(resmodel, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)    

#Test the accuracy with test data
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = resmodel(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

from sklearn.metrics import f1_score
# Iterate over data.
y_true, y_pred = [], []
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        #outputs = model(inputs)
        predicted_outputs = resmodel(inputs)
        _, predicted = torch.max(predicted_outputs, 1)
        total += labels.size(0)
        #print(total)
        correct += (predicted == labels).sum().item()
        #print(correct)
        #f1 score
        temp_true=labels.cpu().data.numpy()
        temp_pred=predicted.cpu().data.numpy()
        y_true+=temp_true.tolist()
        y_pred+=temp_pred.tolist()
        
print('F1 Score:')
f1=f1_score(y_true,y_pred, average = 'macro')
print(f1)

resmodel_path='/home/snake/Desktop/imbalanced/resmodel.pth'
torch.save(gmodel.state_dict(), resmodel_path)
resmodel.load_state_dict(torch.load(resmodel_path))


# Load the pre-trained VGG-16 model
vggmodel = models.vgg16(pretrained=True)

# Replace the last layer with a new linear layer that has 10 output units (for 10 classes)
num_features = vggmodel.classifier[-1].in_features
vggmodel.classifier[-1] = nn.Linear(num_features, 8)

# Move the model to a specific device (e.g. GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vggmodel = vggmodel.to(device)

# Instantiate a Cross Entropy Loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Using Adam as the parameter optimizer
optimizer_ft = optim.Adam(vggmodel.parameters(), lr = 0.001, betas=(0.9, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

vggmodel = train_model(vggmodel, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10) 

#Test the accuracy with test data
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = vggmodel(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

from sklearn.metrics import f1_score
# Iterate over data.
y_true, y_pred = [], []
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        #outputs = model(inputs)
        predicted_outputs = vggmodel(inputs)
        _, predicted = torch.max(predicted_outputs, 1)
        total += labels.size(0)
        #print(total)
        correct += (predicted == labels).sum().item()
        #print(correct)
        #f1 score
        temp_true=labels.cpu().data.numpy()
        temp_pred=predicted.cpu().data.numpy()
        y_true+=temp_true.tolist()
        y_pred+=temp_pred.tolist()
        
print('F1 Score:')
f1=f1_score(y_true,y_pred, average = 'macro')
print(f1)

vggmodel_path='/home/snake/Desktop/imbalanced/resmodel.pth'
torch.save(vggmodel.state_dict(), vggmodel_path)
vggmodel.load_state_dict(torch.load(vggmodel_path))


# Load the pre-trained AlexNet model
alexnet_model = models.alexnet(pretrained=True)

# Replace the last layer with a new linear layer that has 5 output units (for 5 classes)
num_features = alexnet_model.classifier[-1].in_features
alexnet_model.classifier[-1] = nn.Linear(num_features, 8)

# Move the model to a specific device (e.g. GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alexnet_model = alexnet_model.to(device)

# Instantiate a Cross Entropy Loss function for multi-class classification
criterion = nn.CrossEntropyLoss()

# Using Adam as the parameter optimizer
optimizer_ft = optim.Adam(alexnet_model.parameters(), lr = 0.001, betas=(0.9, 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


alexnet_model = train_model(alexnet_model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10) 


#Test the accuracy with test data
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = alexnet_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))



from sklearn.metrics import f1_score
# Iterate over data.
y_true, y_pred = [], []
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['TEST']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        #outputs = model(inputs)
        predicted_outputs = alexnet_model(inputs)
        _, predicted = torch.max(predicted_outputs, 1)
        total += labels.size(0)
        #print(total)
        correct += (predicted == labels).sum().item()
        #print(correct)
        #f1 score
        temp_true=labels.cpu().data.numpy()
        temp_pred=predicted.cpu().data.numpy()
        y_true+=temp_true.tolist()
        y_pred+=temp_pred.tolist()
        # Iterate through the batch and compare the predicted labels to the true labels
        
print('F1 Score:')
f1=f1_score(y_true,y_pred, average = 'macro')
print(f1)



alexnet_model_path='/home/snake/Desktop/imbalanced/alexnet_model.pth'
torch.save(alexnet_model.state_dict(), alexnet_model_path)
alexnet_model.load_state_dict(torch.load(alexnet_model_path))


# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
train_data = datasets.ImageFolder(root="/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/TRAIN", transform=transform)
test_data = datasets.ImageFolder(root="/mnt/nas4/Zhansaya_affectnet/train_set/BALANCED_FINAL/TEST", transform=transform)

# Define the AdaBoost classifier
classifier = AdaBoostClassifier(n_estimators=50)

# Define the PyTorch dataloaders
train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

# Define the neural network
class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 8)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


# Define the training function
def train(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Define the evaluation function
def evaluate(model, criterion, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = EmotionClassifier().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(10):
    train_loss = train(model, criterion, optimizer, train_loader)
    test_acc = evaluate(model, criterion, test_loader)
    print("Epoch {}, Train Loss: {:.4f}, Test Accuracy: {:.4f}".format(epoch+1, train_loss, test_acc))

# Train the AdaBoost classifier on the last layer features
X_train = []
y_train = []
for inputs, labels in train_loader:
    inputs = inputs.to(device)
    features = model.features(inputs).view(inputs.size(0), -1).detach().cpu().numpy()
    X_train.append(features)
    y_train.append(labels.numpy())
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
classifier.fit(X_train, y_train)

# Evaluate the AdaBoost classifier on the testing dataset
X_test = []
y_test = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        features = model.features(inputs).view(inputs.size(0), -1).detach().cpu().numpy()
        X_test.append(features)
        y_test.append(labels.numpy())
X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)
test_acc = classifier.score(X_test, y_test)
print("Test Accuracy (AdaBoost): {:.4f}".format(test_acc))


#Test the accuracy with test data
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))


from sklearn.metrics import f1_score
# Iterate over data.
y_true, y_pred = [], []
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        #outputs = model(inputs)
        predicted_outputs = model(inputs)
        _, predicted = torch.max(predicted_outputs, 1)
        total += labels.size(0)
        #print(total)
        correct += (predicted == labels).sum().item()
        #print(correct)
        #f1 score
        temp_true=labels.cpu().data.numpy()
        temp_pred=predicted.cpu().data.numpy()
        y_true+=temp_true.tolist()
        y_pred+=temp_pred.tolist()
        # Iterate through the batch and compare the predicted labels to the true labels
        
print('F1 Score:')
f1=f1_score(y_true,y_pred, average = 'macro')
print(f1)


adaboost_model_path='/home/snake/Desktop/imbalanced/adaboost_model.pth'
torch.save(adaboost_model.state_dict(), adaboost_model_path)
model.load_state_dict(torch.load(alexnet_model_path))

