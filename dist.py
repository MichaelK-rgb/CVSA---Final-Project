import os

directory = '/datashare/APAS/'

gesture_folders = ['transcriptions_gestures', 'transcriptions_tools_right_new', 'transcriptions_tools_left_new']

gesture_data_dict = {}

for folder in gesture_folders:
    folder_path = os.path.join(directory, folder)
    gesture_data_dict[folder] = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                gesture_data = f.readlines()
                gesture_dict = {}
                for line in gesture_data:
                    start, end, gesture_name = line.strip().split()
                    gesture_dict[(int(start), int(end))] = gesture_name
                gesture_data_dict[folder][filename] = gesture_dict


my_dict = {
    ("T0", "T0"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T0", "T1"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T0", "T2"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T0", "T3"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T1", "T0"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T1", "T1"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T1", "T2"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T1", "T3"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T2", "T0"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T2", "T1"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T2", "T2"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T2", "T3"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T3", "T0"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T3", "T1"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T3", "T2"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}, 
    ("T3", "T3"): {"G0":0, "G1":0,"G2":0,"G3":0,"G4":0,"G5":0}
}

for vid in gesture_data_dict['transcriptions_gestures'].keys():
  g = gesture_data_dict['transcriptions_gestures'][vid]
  r = gesture_data_dict['transcriptions_tools_right_new'][vid]
  l = gesture_data_dict['transcriptions_tools_left_new'][vid]
  for i in g.keys():
    f_f = i[0]
    l_f = i[1]
    g_f = g[i]
    for j in r.keys():
      if (j[0] >= f_f) and (j[1] <= l_f):
        r_t = r[j]
        break
    for j in l.keys():
      if (j[0] >= f_f) and (j[1] <= l_f):
        l_t = l[j]
        break
    my_dict[(l_t, r_t)][g_f] += l_f - f_f

import numpy as np

def get_probability_distributions(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        total = sum(value.values())
        if total == 0:
          probabilities = {k: v for k, v in value.items()}
        else:
          probabilities = {k: v/total for k, v in value.items()}
        output_dict[key] = probabilities
    return output_dict
    
print(get_probability_distributions(my_dict))                                
            