# ComputerVision_in_OR
Open Surgery object detection, tool usage classification

# Introduction 
Tool detection and tool usage classification in OR footage are crucial requirements for future advancements in smart OR technologies such as AI driven robotic assistants as well as effective skill assessment of medical students (Goldbraikh, et al). In this project, we use the suturing skill assessment footage that was used in the 2022 study to train our own CV model for detection of hand + tool combinations and classification of tool usage.

![image](https://user-images.githubusercontent.com/65919086/209307681-56749bda-0356-4f66-9298-2353fe645db8.png)  
         Classification labels: [Right_Needle_driver, Left_Forceps]
         
# Data       
After several augmentations and 'horizontal flip', the class distribution of the dataset is as follows:
![image](https://user-images.githubusercontent.com/65919086/209871355-0f8f0ab5-cd4a-4dd2-b721-53352f359e65.png)

# Experiments 
We focused on 2 tasks, showcasing the tradeoff between performance and speed/ complexity: 
-	Creating a model to classify tool usage in *Real-Time*
-	Creating a model to evaluate pre-recorded video with focus on *Accuracy*  

For this purpose, we experimented with various YOLOv5 models: nano, small and XLarge. finally choosing YOLOv5s for the Real-Time task, and YOLOv5X for the pre-recorded evaluation task.

Class	Instances	P	Recall	mAP50	mAP 25	mAP 75	mAP 50-95
all	796	0.943	0.932	0.957	0.959	0.936	0.835
0	48	0.964	0.896	0.94	0.94	0.895	0.773
1	48	0.896	0.897	0.948	0.948	0.887	0.798
2	150	0.959	0.943	0.947	0.957	0.929	0.843
3	150	0.958	0.919	0.948	0.957	0.93	0.848
4	62	0.94	0.952	0.979	0.979	0.979	0.859
5	62	0.958	0.935	0.978	0.978	0.978	0.855
6	138	0.936	0.957	0.961	0.961	0.948	0.862
7	138	0.93	0.956	0.953	0.953	0.942	0.841
