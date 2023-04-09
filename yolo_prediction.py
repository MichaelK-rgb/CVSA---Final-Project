import torch
import cv2
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
import os
from tqdm import tqdm

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local', force_reload=True)
model.cuda()
model.max_det = 2
res = []
def get_tool_usage(name):
    if "Scissors" in name:
        return "T3"
    elif "Needle_driver" in name:
        return "T1"
    elif "Forceps" in name:
        return "T2"
    else:
        return "T0"

classes = ["Right_Scissors",
           "Left_Scissors",
           "Right_Needle_driver",
           "Left_Needle_driver",
           "Right_Forceps",
           "Left_Forceps",
           "Right_Empty",
           "Left_Empty",
           ]
tool_usage = {"T0": "No tool in hand",
              "T1": "Needle_driver",
              "T2": "Forceps",
              "T3": "Scissors"}
tool_usage_lst = ["No tool in hand", "Needle_driver", "Forceps", "Scissors"]
classes_right = {"Right_Scissors": [],
                 "Right_Needle_driver": [],
                 "Right_Forceps": [],
                 "Right_Empty": []
                 }
classes_left = {"Left_Scissors": [],
                "Left_Needle_driver": [],
                "Left_Forceps": [],
                "Left_Empty": []}
model.names = {x: y for x, y in enumerate(classes)}
image_folder = f'/datashare/APAS/frames/P016_balloon1_top'
for filename in tqdm(os.listdir(image_folder)):
    img_path = os.path.join(image_folder, filename)
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(im_rgb)
        classes_detect = results.pandas().xyxy[0]["name"].values
        cv2.waitKey(0)
        for c in classes_detect:
            if c in classes_right.keys():
                for i in classes_right.keys():
                    if c == i:
                        classes_right[i].append(1)
                    else:
                        classes_right[i].append(0)
        
            elif c in classes_left.keys():
                for i in classes_left.keys():
                    if c == i:
                        classes_left[i].append(1)
                    else:
                        classes_left[i].append(0)

        last_k = 15
        classes_right_val = {x: sum(classes_right[x][-last_k:]) for x in classes_right.keys()}
        classes_left_val = {x: sum(classes_left[x][-last_k:]) for x in classes_left.keys()}
        right_tool = max(classes_right_val, key=classes_right_val.get)
        left_tool = max(classes_left_val, key=classes_left_val.get)
        res.append((get_tool_usage(left_tool), get_tool_usage(right_tool)))
        
import pickle

# Open a file for writing in binary mode
with open("P016_balloon1_top.pickle", "wb") as f:
    # Use the pickle module to dump the classes list to the file
    pickle.dump(res, f)
