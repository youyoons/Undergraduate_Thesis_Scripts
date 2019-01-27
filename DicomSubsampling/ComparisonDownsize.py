import pydicom
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import png
from glob import glob

try:
    import cPickle
except ImportError:
    import pickle as cPickle

#Function: takes in the path of a series and returns a 3D Numpy Array
def create_3d_dicom(series_path):
    dicom_filenames = sorted(os.listdir(series_path))
    dicom_files = [file_name for file_name in dicom_filenames if file_name[-4:]==".dcm"]
    print(dicom_files)
    
    slice_num = 0

    ds = pydicom.dcmread(series_path + dicom_files[0])
    image_2d = ds.pixel_array.astype(float)
    size_2d = np.shape(image_2d)
    dicom_3d = np.zeros((size_2d[0],size_2d[1],len(dicom_files)))

    for file_name in dicom_files:
        ds = pydicom.dcmread(series_path + file_name)
    
        image_2d = ds.pixel_array.astype(float)
        image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    
        image_2d_scaled = np.uint8(image_2d_scaled)
        
        dicom_3d[:,:,slice_num] = image_2d_scaled
    
        slice_num += 1
    
    print(dicom_3d.shape)
    
    return dicom_3d

def volumeOfInterest(reference):
    reference_dim = np.shape(reference)
    
    #Frontal
    fig = plt.figure(figsize = (24,8))
    plt.gray()

    for i in range(1,6):    
        fig.add_subplot(1,5,i)
        plt.imshow(reference[:,:,int(reference_dim[2]/6*i)])
    
    #Side 
    fig2 = plt.figure(figsize = (20,10))
    plt.gray()

    for i in range(1,6):    
        fig2.add_subplot(1,5,i)
        plt.imshow(reference[:,int(reference_dim[1]/6*i),:])
    
    plt.show()
    
    return reference

def downsize_2x(dicom_3d,narrow_3rd=True):
    #Get dimensions of original dicom image
    dicom_3d_dim = np.shape(dicom_3d)
    
    if narrow_3rd:
        #Set downsized dicom image to be half for each side (1/8 the volume)
        dicom_3d_downsized = np.zeros((dicom_3d_dim[0]//2,dicom_3d_dim[1]//2,dicom_3d_dim[2]//2))
    else:
        #Set downsized dicom image to be half for the non-narrow sides (1/4 the volume)
        dicom_3d_downsized = np.zeros((dicom_3d_dim[0]//2,dicom_3d_dim[1]//2,dicom_3d_dim[2]))
        
    for i in range(dicom_3d_downsized.shape[0]):
        for j in range(dicom_3d_downsized.shape[1]):
            for k in range(dicom_3d_downsized.shape[2]):
                if narrow_3rd:
                    dicom_3d_downsized[i,j,k] = (dicom_3d[2*i,2*j,2*k] + dicom_3d[2*i+1,2*j,2*k] + dicom_3d[2*i,2*j+1,2*k] + dicom_3d[2*i+1,2*j+1,2*k] + dicom_3d[2*i,2*j,2*k+1] + dicom_3d[2*i+1,2*j,2*k+1] + dicom_3d[2*i,2*j+1,2*k+1] + dicom_3d[2*i+1,2*j+1,2*k+1])/8
                else:
                    dicom_3d_downsized[i,j,k] = (dicom_3d[2*i,2*j,k] + dicom_3d[2*i+1,2*j,k] + dicom_3d[2*i,2*j+1,k] + dicom_3d[2*i+1,2*j+1,k])/4
    
    #Rounding the Downsized 
    dicom_3d_downsized = np.rint(dicom_3d_downsized)
    
    return dicom_3d_downsized


ac_num = "4697342"

raw_dicom_3d = create_3d_dicom("/home/youy/Documents/Spine/ProcessedData_y/" + ac_num + "/1.2.840.113619.2.5.20242221.1153138343.602.5/")
volumeOfInterest(raw_dicom_3d)

cPickle.dump(raw_dicom_3d, open("no_fractures/dicom_3d_" + ac_num + ".pkl", "wb"), protocol = 2)

dwn2x_dicom_3d = downsize_2x(raw_dicom_3d)
volumeOfInterest(dwn2x_dicom_3d)

dwn4x_dicom_3d = downsize_2x(dwn2x_dicom_3d)
volumeOfInterest(dwn4x_dicom_3d)

dwn8x_dicom_3d = downsize_2x(dwn4x_dicom_3d)
volumeOfInterest(dwn8x_dicom_3d)
