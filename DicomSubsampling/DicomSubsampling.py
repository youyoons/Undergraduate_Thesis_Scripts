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

#===================================================================================================
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
    
        image_2d = ds.pixel_array.astype(float)
        image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    
        image_2d_scaled = np.uint8(image_2d_scaled)
        
        dicom_3d[:,:,slice_num] = image_2d_scaled
    
        slice_num += 1
    
    print(dicom_3d.shape)
    
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

#Set save_orig to True if you want to save original full-sized DICOM file
def generate_pkl(series_path, ac_num, save_orig=False, from_scratch=True):
    if from_scratch:
        raw_dicom_3d = create_3d_dicom(series_path)
    else:
        try:
            raw_dicom_3d = cPickle.load(open("no_fractures/dicom_3d_" + ac_num + ".pkl", "rb"))
        except:
            raw_dicom_3d = cPickle.load(open("no_fractures/dicom_3d_" + ac_num + ".pkl","rb"),encoding = 'latin1')
    
    if save_orig:
        cPickle.dump(raw_dicom_3d, open("no_fractures/dicom_3d_" + ac_num + ".pkl", "wb"))
    
    raw_dicom_3d_dim = np.shape(raw_dicom_3d)    

    #Remove first slice as it generally contains noisy, non-sensical data
    dicom_3d = raw_dicom_3d[:,:,1:((raw_dicom_3d_dim[2]-1)//4)*4+1]
    dicom_3d_dim = np.shape(dicom_3d)
    print("Dimensions of Full-Sized DICOM: ", dicom_3d_dim)
    
    #True to downsize the z axis by 2x
    dicom_3d_downsized = downsize_2x(dicom_3d,True)
    
    #Look at z axis length to see whether to narrow z-axis by 4x
    if dicom_3d_dim[2] > 150:
        dicom_3d_4x_downsized = downsize_2x(dicom_3d_downsized,True)
    else:
        dicom_3d_4x_downsized = downsize_2x(dicom_3d_downsized,False)
    
    dicom_3d_4x_dim = np.shape(dicom_3d_4x_downsized)
    print("Dimension of Down-Sized DICOM: ", dicom_3d_4x_dim)
    
    #Save the final downsized image as pkl file
    cPickle.dump(dicom_3d_4x_downsized, open("no_fractures/dicom_3d_" + ac_num + "_dwn4x.pkl","wb"))
    
    return None


if __name__ == '__main__':
    os.chdir("C:\\Users\\yoons\\Documents\\4th Year Semester 2\\ESC499 - Thesis\\Undergraduate_Thesis_Scripts\\DicomSubsampling")

    #Set create_pkl to true if we are interested in creating pkl files from scratch
    create_pkl = False #If we want to create a downsized pkl
    create_pkl_from_scratch= False #If we want to do this from raw .dcm files or from a pre-existing full-sized pkl
    vol_of_interest = True #If we want to create a reference.pkl
    plot_downsized = False #If we want to plot the downsized 4x volume



#===================================================================================================
    #Want to create downsized pickle object file
    if create_pkl:
        if create_pkl_from_scratch:
            base_series_path = "/home/youy/Documents/Spine/ProcessedData_y/"
            
            print("***CONVERT FULL-SIZED DICOM (.DCM) TO SUBSAMPLED DICOM PKL***")
            
            #Get all series that are processed
            series_paths = glob(base_series_path + "/*/*/")
            
            #Generate downsized pkl files for all series
            for series_path in series_paths:
                path_one_level_up = os.path.dirname(os.path.dirname(series_path))
                ac_num = path_one_level_up.split("/")[-1]
                
                #Only create pkl file if it does not exist
                if os.path.isfile("no_fractures/dicom_3d_" + ac_num + "_dwn4x.pkl") == False:
                    print("ACCESSION NUMBER: ",ac_num)
                    generate_pkl(series_path, ac_num, False, True)
                
        #This is used when we want to make a pre-existing full sized pkl to a downsized one (edge case)
        else:
            ac_num = "9128577"
            print("***CONVERT FULL-SIZED DICOM PKL TO SUBSAMPLED DICOM PKL***")
            print("ACCESSION NUMBER: ", ac_num)
            
            #No series path needed, ac_num provided by user, no save_orig, not from_scratch
            generate_pkl(None, ac_num, False, False)
                    
 

    
#===================================================================================================
    #Getting Reference Image for 3D GHT
    if vol_of_interest:

        ac_num = "7396455"
        x = 48
        y = 54
        z = 17
        x1 = x - 18
        x2 = x + 18
        y1 = y - 12
        y2 = y + 12
        z1 = z - 9
        z2 = z + 9
        
        try:
            
            try:
                downsized_dicom = cPickle.load(open("no_fractures/dicom_3d_" + ac_num + "_dwn4x.pkl", "rb"))
            except:
                downsized_dicom = cPickle.load(open("no_fractures/dicom_3d_" + ac_num + "_dwn4x.pkl","rb"),encoding = 'latin1')
    
            #Get volume of interest
            reference = volumeOfInterest(downsized_dicom[x1:x2,y1:y2,z1:z2])
        
            #cPickle.dump(reference, open("no_fractures/dicom_3d_" + ac_num + "_reference.pkl", "wb"))
        
        except:
            print("Cannot find " + ac_num)
        



#===================================================================================================
    #Getting the plot of necessary 
    if plot_downsized:
    
        ac_num = "5056218"
        
        try:
            dicom_plot = cPickle.load(open("no_fractures/dicom_3d_" + ac_num + "_dwn4x.pkl", "rb"))
        except:
            dicom_plot = cPickle.load(open("no_fractures/dicom_3d_" + ac_num + "_dwn4x.pkl","rb"),encoding = 'latin1')
        
        dicom_plot_dim = np.shape(dicom_plot)
        
        #cPickle.dump(dicom_plot[:,:,1:],open("no_fractures/dicom_3d_" + ac_num + "_dwn4x.pkl","wb"))
        
        #Frontal
        fig = plt.figure()
        plt.gray()
    
        for i in range(1,6):    
            fig.add_subplot(1,5,i)
            plt.imshow(dicom_plot[:,:,int(dicom_plot_dim[2]/6*i)])
        
        #Side 
        fig2 = plt.figure()
        plt.gray()
    
        for i in range(1,6):    
            fig2.add_subplot(1,5,i)
            plt.imshow(dicom_plot[:,int(dicom_plot_dim[1]/6*i),:])
        
        plt.show() 
        
