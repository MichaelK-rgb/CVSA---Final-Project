import torch
import cv2
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from statistics import mean
import time

yolo_model = 'best_s'
video = 'P026_tissue1'
model = torch.hub.load('yolov5', 'custom', path=yolo_model + '.pt', source='local', force_reload=True)
model.cuda()
model.max_det = 2
cap = cv2.VideoCapture(video + ".wmv")
last_frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
df_right = pd.read_csv(video + "_right.txt", header=None, sep=" ")
df_left = pd.read_csv(video + "_left.txt", header=None, sep=" ")
df_right["dir"] = "right"
df_left["dir"] = "left"
frames = [df_right, df_left]
result = pd.concat(frames)
counter = -1
conf_threshold = 0.85 if yolo_model == 'best_s' else 0.8

vid_result = cv2.VideoWriter(f'{video}labeled.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             30, (640, 640))

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
           "Left_Empty"]
preds_right = []
labels_right = []
preds_left = []
labels_left = []
confidence_left = []
confidence_right = []

model.names = {x: y for x, y in enumerate(classes)}
# Read until video is completed
st = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        counter += 1
        if counter == int(last_frame_num) - 1:
            break
        print(counter)
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
        classes_confidence = results.pandas().xyxy[0]['confidence'].values
        try:
            confidence_left.append(classes_confidence[1])
        except:
            pass
        try:
            confidence_right.append(classes_confidence[0])
        except:
            pass

        for c in classes_detect:
            index = np.where(classes_detect == c)[0][0]
            if classes_confidence[index] > conf_threshold:
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
            else:
                if c in classes_right.keys():
                    for i in classes_right.keys():
                        if classes_right[i][-1] == 1:
                            classes_right[i].append(1)
                        else:
                            classes_right[i].append(0)

                if c in classes_left.keys():
                    for i in classes_left.keys():
                        if classes_left[i][-1] == 1:
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
        # cv2.imshow('Frame', frame)
        vid_result.write(frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
vid_result.release()
cv2.destroyAllWindows()
et = time.time()
print('Execution time:', (et - st), 'seconds')

labels_right_num = [tool_usage_lst.index(i) for i in labels_right]
labels_left_num = [tool_usage_lst.index(i) for i in labels_left]
preds_right_num = []
preds_left_num = []

for i in preds_right:
    if i[-2:] == "rs":
        preds_right_num.append(3)
    if i[-2:] == "er":
        preds_right_num.append(1)
    if i[-2:] == "ps":
        preds_right_num.append(2)
    if i[-2:] == "ty":
        preds_right_num.append(0)

for i in preds_left:
    if i[-2:] == "rs":
        preds_left_num.append(3)
    if i[-2:] == "er":
        preds_left_num.append(1)
    if i[-2:] == "ps":
        preds_left_num.append(2)
    if i[-2:] == "ty":
        preds_left_num.append(0)

target_names = tool_usage.values()

print("right hand")
print(classification_report(labels_right_num, preds_right_num))
print(f"F1 score: {f1_score(labels_right_num, preds_right_num, average='macro')}")
print(f"Acc score: {accuracy_score(labels_right_num, preds_right_num)}")
print(f'avg confidence level: {mean(confidence_right)}')

print("left hand")
print(classification_report(labels_left_num, preds_left_num))
print(f"F1 score: {f1_score(labels_left_num, preds_left_num, average='macro')}")
print(f"Acc score: {accuracy_score(labels_left_num, preds_left_num)}")
print(f'avg confidence level: {mean(confidence_left)}')
