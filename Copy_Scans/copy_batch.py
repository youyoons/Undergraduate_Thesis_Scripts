import shutil
import os

filename = open("path_frac.log","r+")

base_path = "/home/youy/Documents/Spine/batch_test2/"

for line in filename:
    print(line[:-1])
    fromDirectory = line[:-1]
    if os.path.isdir(fromDirectory):
        shutil.copytree(fromDirectory,toDirectory)
    else:
        shutil.copy2(fromDirectory,toDirectory)
    
