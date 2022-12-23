import torch
import cv2
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
import sys

image = sys.argv[1]
video = 'P023_tissue2'
model = torch.hub.load('yolov5', 'custom', path='small.pt', source='local', force_reload=True)
# model.cuda()
model.max_det = 2
cap = cv2.VideoCapture(video + ".wmv")
df_right = pd.read_csv(video + "_right.txt", header=None, sep=" ")
df_left = pd.read_csv(video + "_left.txt", header=None, sep=" ")
df_right["dir"] = "right"
df_left["dir"] = "left"
frames = [df_right, df_left]
result = pd.concat(frames)
counter = -1
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
classes = ["Right_Scissors",
           "Left_Scissors",
           "Right_Needle_driver",
           "Left_Needle_driver",
           "Right_Forceps",
           "Left_Forceps",
           "Right_Empty",
           "Left_Empty",
           ]
preds_right = []
labels_right = []
preds_left = []
labels_left = []

model.names = {x: y for x, y in enumerate(classes)}
# Read until video is completed
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         counter += 1
#         # if counter < 2430:
#         #     continue
#         print(counter)
frame = cv2.imread(image)
filter_1 = result[0] < counter
filter_2 = result[1] > counter
tmp = result.where(filter_1)
tmp = tmp.where(filter_2)
tmp = tmp.dropna()
rel = tmp[[2, "dir"]]
frame = cv2.resize(frame, (640, 640))
im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = model(im_rgb)
classes_detect_x = results.pandas().xyxy[0]["xmin"].values
classes_detect = results.pandas().xyxy[0]["name"].values

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
# frame = cv2.resize(frame, (640, 720))
font = cv2.FONT_HERSHEY_DUPLEX
org = (25, 25)
fontScale = 0.7
color = (255, 0, 125)
thickness = 2
frame = results.render()[0]
frame = cv2.putText(frame, right_tool + (" " * (33 - len(right_tool))) + left_tool, org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

# "                      "
if len(rel) == 2:
    org = (25, 50)
    color = (0, 255, 0)
    ground_truth = tool_usage[rel.iloc[0][2]] + (" " * (33 - len(tool_usage[rel.iloc[0][2]]))) + tool_usage[
        rel.iloc[1][2]]
    frame = cv2.putText(frame, ground_truth, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
if len(rel) == 2 and len(classes_detect) == 2:
    if classes_detect[0][0] == classes_detect[1][0]:
        if classes_detect_x[0] < classes_detect_x[1]:
            preds_left.append(classes_detect[1])
            preds_right.append(classes_detect[0])
        else:
            preds_left.append(classes_detect[0])
            preds_right.append(classes_detect[1])

    else:
        for c in classes_detect:
            if c in classes_right.keys():
                preds_right.append(c)
            if c in classes_left.keys():
                preds_left.append(c)
    labels_right.append(tool_usage[rel.iloc[0][2]])
    labels_left.append(tool_usage[rel.iloc[1][2]])
cv2.imshow('Frame', frame)
# Filename
filename = 'predicted.jpg'
cv2.imwrite(filename, frame)

