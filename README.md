# Emotion-Classification-Head-Pose-Orientation
## Setup
- Please, download pre-trained models' weights from https://www.kaggle.com/datasets/zhansayasovetbek/emotion-classification-pre-trained-models-weights to train the model.
- You can download datasets from https://www.kaggle.com/datasets/zhansayasovetbek/emotion-classification-head-direction-balanced
## Install
- https://docs.anaconda.com/free/anaconda/install/windows/
- https://pytorch.org/get-started/locally/
- Warning! Install Pytorch with CUDA.
## Introduction
- Facial emotion recognition has received increasing attention in recent years due to its
potential applications in various fields such as human-computer interaction, security,
and healthcare. In this context, the orientation of a face has been identified as an
important factor affecting the accuracy of facial emotion recognition.
- Two methodological approaches were used in this research: the baseline model and
the proposed model. All two models classify the face orientation directions and facial
emotions. The models will use Hopenet to identify head pose direction angles such
as pitch, yaw, and roll then to determine one of the directions, namely, forward, left,
right, up, and down.
- Pre-trained models such as MobileNetV3-small, ResNet-18, GoogleNet, and others
will be used to classify emotions and find the connection between facial emotion
classification and head pose orientation.
## Datasets
- AffectNet(440K) images classified into ("happy", "sad", "surprise", "fear", "disgust", "anger", "contempt", "neutral") emotions.
- The Pointing’04 (15,000 images of people’s faces)
from various perspectives
- The AFLW2000-3D (2,000 3D 18 annotated face photos)
## Models
There are two models: Baseline and Proposed. All of them used pre-trained models.
### Pre-trained models
- HopeNet (for Head Pose Direction angles)
- MobileNetV3-small
- Googlenet
- ResNet-18
- VGG-16
- Alexnet
- AdaBoost
- Simple NN
## Methodology
![image](https://github.com/zhansayaszh/Emotion-Classification-Head-Pose-Orientation/assets/28733943/f74fe04a-b8fb-4ec5-a234-b86aafd079e3)
## Results
### Baseline Model Results
![image](https://github.com/zhansayaszh/Emotion-Classification-Head-Pose-Orientation/assets/28733943/1c7dbea3-1740-4c69-9530-a4893386bde4)
![image](https://github.com/zhansayaszh/Emotion-Classification-Head-Pose-Orientation/assets/28733943/0af9b655-6703-4ce9-a09d-1569b4192e83)

### Proposed Model Results
![image](https://github.com/zhansayaszh/Emotion-Classification-Head-Pose-Orientation/assets/28733943/bebea7e1-ec3c-4b1f-a336-5ec9b0cd3ed0)
![image](https://github.com/zhansayaszh/Emotion-Classification-Head-Pose-Orientation/assets/28733943/dcd4d1c1-0905-4b0e-8ab5-0a05137ea3b4)

## Conclusion
![image](https://github.com/zhansayaszh/Emotion-Classification-Head-Pose-Orientation/assets/28733943/54a52ce8-fe34-4ee0-a237-2ca65bc46d52)

Even though the percentage distribution of directions is almost the same, it can be
seen that some directions suppress others, and could predict emotion.
![image](https://github.com/zhansayaszh/Emotion-Classification-Head-Pose-Orientation/assets/28733943/0e869e58-5b53-40fa-897f-4490298abfb0)







