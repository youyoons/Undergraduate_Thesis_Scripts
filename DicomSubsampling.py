import pydicom
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

os.getcwd()

patient_path = "/home/youy/Documents/Spine/RawData/1.2.840.113619.2.25.4.1260075.1281918795.857/1.2.840.113619.2.25.4.1260075.1281918796.268/"

dicom_filenames = sorted(os.listdir(patient_path))
dicom_files = [file_name for file_name in dicom_filenames if file_name[-4:]==".dcm"]
print(dicom_files)

slice_num = 0
dicom_3d = np.zeros((512,512,len(dicom_files)))

for file_name in dicom_files:
    ds = pydicom.dcmread(patient_path + file_name)

    #shape = ds.pixel_array.shape
    #print(shape)

    image_2d = ds.pixel_array.astype(float)
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0

    image_2d_scaled = np.uint8(image_2d_scaled)
    
    dicom_3d[:,:,slice_num] = image_2d_scaled

    slice_num += 1

#print(image_2d_scaled.shape)
#print(image_2d_scaled[100][100])
print(dicom_3d.shape)
print(dicom_3d[100,100,30])

dicom_3d_downsized = np.zeros((256,256,len(dicom_files)))

for k in range(dicom_3d.shape[2]):
    for i in range(dicom_3d.shape[0]-1):
        for j in range(dicom_3d.shape[1]-1):
            dicom_3d_downsized[i,j,k] = (dicom_3d[2*i,2*j,k] + dicom_3d[2*i+1,2*j,k] + dicom_3d[2*i,2*j+1,k] + dicom_3d[2*i+1,2*j+1,k])/4



imgplot = plt.imshow(dicom_3d[:,:,20])
plt.show()

#plt.imshow(image_2d_scaled)
#plt.show()
