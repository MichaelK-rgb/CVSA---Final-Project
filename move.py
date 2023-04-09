import os
import shutil

# Set the paths to the folders you want to copy
folder1_path = "/datashare/APAS/frames/P016_balloon1_side"
folder2_path = "/datashare/APAS/frames/P020_balloon1_side"
folder3_path = "/datashare/APAS/frames/P022_balloon1_side"

# Get the directory where the script is located
script_dir = "/home/student/MS-TCN++"

# Copy the folders to the script directory
print("before")
shutil.copytree(folder1_path, os.path.join(script_dir, "folder1"))
print("finish 1")
shutil.copytree(folder2_path, os.path.join(script_dir, "folder2"))
print("finish 2")
shutil.copytree(folder3_path, os.path.join(script_dir, "folder3"))