import pydicom
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import png

try:
    import cPickle
except ImportError:
    import pickle as cPickle

def create_dicom_png(series_path):
    dicom_filenames = sorted(os.listdir(series_path))
    dicom_files = [file_name for file_name in dicom_filenames if file_name[-4:] == ".dcm"]

    destination = '/home/youy/Documents/DICOM_PNG/'

    for file_name in dicom_files:
        ds = pydicom.dcmread(series_path + file_name)
        shape = ds.pixel_array.shape

        image_2d = ds.pixel_array.astype(float)
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

        image_2d_scaled = np.uint8(image_2d_scaled)

        with open(destination + file_name[:-4] + ".png",'wb') as png_file:
            w = png.Writer(shape[1],shape[0], greyscale=True)
            w.write(png_file, image_2d_scaled)

    return None


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
    
        #shape = ds.pixel_array.shape
        #print(shape)
    
        image_2d = ds.pixel_array.astype(float)
        image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    
        image_2d_scaled = np.uint8(image_2d_scaled)
        
        dicom_3d[:,:,slice_num] = image_2d_scaled
    
        slice_num += 1
    
    print(dicom_3d.shape)
    print(dicom_3d[100,100,30])
    
    return dicom_3d

'''
Function: read in a 3D Numpy array and downsize it by 2x (side lengths)
Inputs: dicom_3d is a 3d numpy array
        narrow_3rd is True if we want to downsize the narrow dimension
'''
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

'''
Function: get a dicom image and select a volume of interest
'''
def volumeOfInterest(dicom):
    #c12 = np.zeros((30,30,50))
    sample = dicom[8:44,40:65,19:32]

    return sample



if __name__ == '__main__':
    #Point to a processed series directory
    #series_path = "/home/youy/Documents/Spine/ProcessedData_y/6585553/1.2.840.113619.2.25.4.235810714.7183.1306015929.539/"
    
    #dicom_3d = create_3d_dicom(series_path)

    #cPickle.dump(dicom_3d, open("dicom_3d_sample.pkl", "wb"))
    #

    try:
        dicom_3d = cPickle.load(open("dicom_3d_9183761.pkl", "rb"))
    except:
        dicom_3d = cPickle.load(open("dicom_3d_9183761.pkl","rb"),encoding = 'latin1')
        
        
    dicom_3d_downsized = downsize_2x(dicom_3d,False)
    
    dicom_3d_4x_downsized = downsize_2x(dicom_3d_downsized,True)

    #Get volume of interest
    sample = volumeOfInterest(dicom_3d_4x_downsized)
 
    cPickle.dump(sample, open("dicom_3d_9183761_sample.pkl", "wb"))
    cPickle.dump(dicom_3d_4x_downsized, open("dicom_3d_9183761_dwn4x.pkl","wb"))
    
    fig = plt.figure()
    plt.gray()

    for i in range(13):    
        fig.add_subplot(5,3,i+1)
        plt.imshow(sample[:,:,i])
    
    
    plt.show() 
 
'''
    #Testing to print values
    fig = plt.figure()
    plt.gray()
    fig.add_subplot(3,2,1)
    plt.title('Full Size - Beginning')
    plt.imshow(dicom_3d[:,:,0])
    
    fig.add_subplot(3,2,2)
    plt.title('Full Size - Middle')
    plt.imshow(dicom_3d[:,:,50])
    
    fig.add_subplot(3,2,3)
    plt.title('Downsized 2x - Beginning')
    plt.imshow(dicom_3d_downsized[:,:,0])
    
    fig.add_subplot(3,2,4)
    plt.title('Downsized 2x - Middle')
    plt.imshow(dicom_3d_downsized[:,:,50])
    
    fig.add_subplot(3,2,5)
    plt.title('Downsized 4x - Beginning')
    plt.imshow(dicom_3d_4x_downsized[:,:,0])
    
    fig.add_subplot(3,2,6)
    plt.title('Downsized 4x - Middle')
    plt.imshow(dicom_3d_4x_downsized[:,:,25])
    
    plt.show()
'''


