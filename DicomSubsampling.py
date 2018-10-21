import pydicom
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#Function: takes in the path of a series and returns a 3D Numpy Array
def create_3d_dicom(series_path):
    dicom_filenames = sorted(os.listdir(series_path))
    dicom_files = [file_name for file_name in dicom_filenames if file_name[-4:]==".dcm"]
    print(dicom_files)
    
    slice_num = 0
    dicom_3d = np.zeros((512,512,len(dicom_files)))
    
    for file_name in dicom_files:
        ds = pydicom.dcmread(series_path + file_name)
    
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
    
    return dicom_3d

#Function: read in a 3D Numpy array and downsize it by 2x (side lengths)
def downsize_2x(dicom_3d):
    #Get dimensions of original dicom image
    dicom_3d_dim = np.shape(dicom_3d)
    
    #Set downsized dicom image to be half for each side (1/8 the volume)
    dicom_3d_downsized = np.zeros((dicom_3d_dim[0]//2,dicom_3d_dim[1]//2,dicom_3d_dim[2]//2))
    
    for i in range(dicom_3d_downsized.shape[0]):
        for j in range(dicom_3d_downsized.shape[1]):
            for k in range(dicom_3d_downsized.shape[2]):
                dicom_3d_downsized[i,j,k] = (dicom_3d[2*i,2*j,2*k] + dicom_3d[2*i+1,2*j,2*k] + dicom_3d[2*i,2*j+1,2*k] + dicom_3d[2*i+1,2*j+1,2*k] + dicom_3d[2*i,2*j,2*k+1] + dicom_3d[2*i+1,2*j,2*k+1] + dicom_3d[2*i,2*j+1,2*k+1] + dicom_3d[2*i+1,2*j+1,2*k+1])/8
    
    
    #Rounding the Downsized 
    dicom_3d_downsized = np.rint(dicom_3d_downsized)
    
    return dicom_3d_downsized




if __name__ == '__main__':
    os.getcwd()
    
    #Point to a processed series directory
    series_path = "/home/youy/Documents/Spine/RawData/1.2.840.113619.2.25.4.1260075.1281918795.857/1.2.840.113619.2.25.4.1260075.1281918796.268/"
    
    dicom_3d = create_3d_dicom(series_path)
    
    #Testing with 3D Images
    #np.random.seed(29)
    #dicom_3d = np.random.randint(0, 256, size=(8,8,8))

    
    dicom_3d_downsized = downsize_2x(dicom_3d)
    
    print(dicom_3d)
    print(dicom_3d_downsized)
    
    '''
    #Testing to print values
    plt.clf() #Clear the Current Figure
    plt.gray() #Set colormap to gray
    
    
    fig = plt.figure()
    fig.add_subplot(2,2,1)
    plt.title('Reference image')
    plt.imshow(reference_image[:,20,:])
    
    fig.add_subplot(2,2,2)
    plt.title('Query image')
    plt.imshow(query_image[:,20,:])
    
    fig.add_subplot(2,2,3)
    plt.title('Accumulator')
    plt.imshow(accumulator[:,20,:])
    
    fig.add_subplot(2,2,4)
    plt.title('Detection')
    plt.imshow(query_image[:,20,:])
    
    plt.show()
    '''