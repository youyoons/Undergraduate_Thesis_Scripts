from distutils.dir_util import copy_tree
import os

#Open the file list generated by get_paths.py
filename = open("path_scans.log","r+")

#Base Destination Directory
#base_path = "/home/youy/Documents/Spine/no_fracture_raw/"
base_path = "/home/youy/Documents/Spine/RawData_y/"

#Go through each file path and copy to destination
for path in filename:
    study_id = path.split("/")[-1]
    
    #Copying from raw data path to local path
    if os.path.exists(os.path.join(base_path,study_id[:-1]+"/")) == False:
        os.mkdir(os.path.join(base_path,study_id[:-1] + "/"))
    
        Source = path[:-1] + "/"
        print("Source: ", Source)
        Dest = os.path.join(base_path,study_id[:-1] + "/")
        print("Destination: ", Dest)
        

        copy_tree(Source,Dest)

