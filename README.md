# Computer Vision in surgical application: Final Project
Computer Vision in surgical application: Final Project

![image](https://user-images.githubusercontent.com/65919086/230781086-75a9d30c-4d22-4d1b-8e30-ff7f7cb054b2.png)

# Introduction 
The growing interest in computer vision and AI tools allows for its implementation in domains such as the operating room, improving the quality of interventional healthcare. Gesture recognition of surgical videos could assist in surgery summarization, progress monitoring, and prediction of surgery steps and duration.
This project explores different action segmentation methods for surgical video. Gesture recognition is a task that requires both frame-wise accuracy as well as temporal memory of the scene. 
Specifically, we will investigate how the MS-TCN++ architecture, a state-of-the-art deep learning model developed by Li et al., can be utilized for this purpose. 

![image](https://user-images.githubusercontent.com/65919086/209307681-56749bda-0356-4f66-9298-2353fe645db8.png)  
         Classification labels: [Right_Needle_driver, Left_Forceps]
         
# Data   
The dataset labels 8 classes of interactions of hand + tools:

{Right_Scissors, Left_Scissors ,Right_Needle_driver, Left_Needle_driver, Right_Forceps, Left_Forceps ,Right_Empty, Left_Empty}

After several augmentations and 'horizontal flip', the class distribution of the dataset is as follows:
![image](https://user-images.githubusercontent.com/65919086/230780796-538767e0-9666-455e-b0c7-e4e79731e5aa.png)

# Model - base model
The MS-TCN++ (multi-scale temporal convolutional network) has proven to be effective in video segmentation, gesture recognition, and other video-based tasks. This model was introduced by Li et al. in their 2020 paper, "Temporal Convolutional Networks for Action Segmentation and Detection." 
MS-TCN ++ is a multi-stage architecture for the temporal action segmentation tasks that consists of 2 stages. The first stage generates an initial prediction that is refined by the next ones. In each stage we stack several layers of dilated temporal convolutions covering a large receptive field with few parameters. 
In addition, the model includes a dual dilated layer that combines both large and small receptive fields. 

# Model - our model

![image](https://user-images.githubusercontent.com/65919086/209872504-b36229ee-15ae-4f65-944e-8e92b8676d17.png)

# Smoothing - sliding window, confidence level threshold
The basic principle of a ‘sliding window’ smoothing means basing the predicted class in each frame on the most predicted label in the previous ‘k’ frames (in the paper k=15). This smoothing is meant to reduce the tool usage misclassification affected by false detections of our object detection model. To experiment with an upgraded approach, a confidence level threshold was applied to the check the current frame’s predicted YOLO bounding boxes. A different threshold was applied for each model size based on the average confidence level that was observed on the test videos (nano-0.81, small-0.85, XL- 0.87). If the confidence for a detected object is higher than the selected threshold, the detected class is added to the window. When it is lower, the previously detected class is appended to the sliding window. This simple modification reduced the misclassification in certain situations where the tool was occluded or held in a previously unseen position.

![image](https://user-images.githubusercontent.com/65919086/209872280-2ef3e072-04be-41f1-963c-409ce66c91da.png)

The Smoothing process using the confidence level threshold and sliding window combination proved to be effective in reducing all scores and misclassification in tool usage and tool usage transitions.

# Reference to YOLOv5 repo
https://github.com/ultralytics/yolov5

