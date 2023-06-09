# Computer Vision in surgical applications: Final Project
Computer Vision in surgical application: Final Project

![image](https://user-images.githubusercontent.com/65919086/230781086-75a9d30c-4d22-4d1b-8e30-ff7f7cb054b2.png)

# Introduction 
The growing interest in computer vision and AI tools allows for its implementation in domains such as the operating room, improving the quality of interventional healthcare. Gesture recognition of surgical videos could assist in surgery summarization, progress monitoring, and prediction of surgery steps and duration.
This project explores different action segmentation methods for surgical video. Gesture recognition is a task that requires both frame-wise accuracy as well as temporal memory of the scene. 
Specifically, we will investigate how the MS-TCN++ architecture, a state-of-the-art deep learning model developed by Li et al., can be utilized for this purpose. 

         
# Data   
This project utilizes data collected by observing and monitoring physicians with varying levels of expertise performing suturing tasks. The data consists of videos captured from two different angles, providing a top-down and side view of the physicians' actions. The input to the MSTCN++ model was visual features of the side-view videos that were extracted through an EfficientNet2-Medium model.
The original input was frames of resolution 224 x 224, and the augmentations performed on the frames were corner cropping, scale jittering and random rotate.

# Model - base model
The MS-TCN++ (multi-scale temporal convolutional network) has proven to be effective in video segmentation, gesture recognition, and other video-based tasks. This model was introduced by Li et al. in their 2020 paper, "Temporal Convolutional Networks for Action Segmentation and Detection." 
MS-TCN ++ is a multi-stage architecture for the temporal action segmentation tasks that consists of 2 stages. The first stage generates an initial prediction that is refined by the next ones. In each stage we stack several layers of dilated temporal convolutions covering a large receptive field with few parameters. 
In addition, the model includes a dual dilated layer that combines both large and small receptive fields. 

# Model - our model

We offer to several modification to the base architecture, namely the addition of an LSTM module that receives as input a downsampled output from the MS-TCN++ model, and the addition of a previously trained YOLOv5 module that will be activated when the confidence level of the MSTCN++ model is lower than a specific threshold. The YOLOv5 model will predict the tools used from the top-angle video of the same operation, in order to improve the gesture prediction. 

![image](https://user-images.githubusercontent.com/65919086/230781167-f8edd932-10ed-436c-8360-0d85c6537ecf.png)

# Results
![image](https://user-images.githubusercontent.com/65919086/230781556-e14a931a-7e37-433c-a722-d36a0930c602.png)

The confidence level threshold + YOLO tool usage prediction in combination with the LSTM module improved F1@10,25,50 and edit scores.

# Reference to MS-TCN++ repo
MS-TCN++: Multi-Stage Temporal Convolutional Network for Action Segmentation (TPAMI 2020)
https://github.com/sj-li/MS-TCN2
# Reference to YOLOv5 repo
https://github.com/ultralytics/yolov5

