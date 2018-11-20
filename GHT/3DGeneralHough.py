import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.misc import imread
from skimage.feature import canny
from scipy.ndimage.filters import sobel
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
from scipy import signal
try:
    import cPickle
except ImportError:
    import pickle as cPickle


def gradient_orientation(image):
    '''
    Calculate the gradient orientation for edge point in the image
    '''
    #scipy.ndimage.sobel
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    dz = sobel(image, axis=1, mode='constant')
    
    #For 3D instead of a single gradient value, we need two angles that define a normal vector
    #Phi is the angle between the positive x-axis to the projection of the normal vector the x-y plane (around +z)
    #Psi is the angle between the positive z-axis to the normal vector
    
    phi = np.arctan2(dy ,dx) * 180 / np.pi
    psi = np.arctan2(np.sqrt(dx*dx + dy*dy), dz) * 180 / np.pi
    

    gradient = np.zeros(image.shape)
    
    return phi, psi

def build_r_table(image, origin):
    '''
    Build the R-table from the given shape image and a reference point
    '''
    
    dx = sobel(image, 0)  # x derivative
    dy = sobel(image, 1)  # y derivative
    dz = sobel(image, 2)  # z derivativeF
    
    mag = np.sqrt(dx*dx + dy*dy + dz*dz)
    mag_norm = mag/np.max(mag)
    
    
    #print(dx[0,0,0], dy[0,0,0], dz[0,0,0], mag_norm[0,0,0])
    
    #print("dx,dy,dz: ", dx.shape,dy.shape,dz.shape)
    
    #Creating edge array the same size as query image
    edges = np.zeros(image.shape, dtype=bool)
    #print("Edge: ", edges.shape)
    
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            for k in range(edges.shape[2]):
                if mag_norm[i,j,k] > 0.3 :# and mag_norm[i,j,k] < 0.9:
                    edges[i,j,k] = True
    



    #Takes (47,40) Edges and calculates the gradients using sobel
    phi, psi = gradient_orientation(edges)
    #print("Phi Dim: ", phi.shape)
    
    r_table = defaultdict(list)
    for (i,j,k),value in np.ndenumerate(edges):
        if value:
            r_table[(int(phi[i,j,k]),int(psi[i,j,k]))].append((origin[0]-i, origin[1]-j, origin[2] - k))
    
    
    return r_table


def sobel_edges_3d(grayImage):
    dx = sobel(grayImage, 0)  # x derivative
    dy = sobel(grayImage, 1)  # y derivative
    dz = sobel(grayImage, 2)  # z derivative
    
    
    #Get magnitude of gradient
    mag = np.sqrt(dx*dx + dy*dy + dz*dz)
    mag_norm = mag/np.max(mag)
    
    #Creating edge array the same size as query image
    edges = np.zeros(grayImage.shape, dtype=bool)
    
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            for k in range(edges.shape[2]):
                if mag_norm[i,j,k] > 0.4 :# and mag_norm[i,j,k] < 0.9:
                    edges[i,j,k] = True      
    
    return edges

def canny_edges_3d(grayImage):
    MIN_CANNY_THRESHOLD = 20
    MAX_CANNY_THRESHOLD = 100
    
    dim = np.shape(grayImage)
    
    edges_x = np.zeros(grayImage.shape, dtype=bool) 
    edges_y = np.zeros(grayImage.shape, dtype=bool) 
    edges_z = np.zeros(grayImage.shape, dtype=bool) 
    edges = np.zeros(grayImage.shape, dtype=bool) 
    
    
    for i in range(dim[0]):
        edges_x[i,:,:] = canny(grayImage[i,:,:], low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD, sigma = 0)
   
    for j in range(dim[1]):
        edges_y[:,j,:] = canny(grayImage[:,j,:], low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD, sigma = 0)
        
    for k in range(dim[2]):
        edges_z[:,:,k] = canny(grayImage[:,:,k], low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD, sigma = 0)
    
    
   # edges = canny(grayImage, low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD)
    for i in range(dim[0]):
        for j in range(dim[1]):
            for k in range(dim[2]):
                #edges[i,j,k] = (edges_x[i,j,k] and edges_y[i,j,k]) or (edges_x[i,j,k] and edges_z[i,j,k]) or (edges_y[i,j,k] and edges_z[i,j,k])
                edges[i,j,k] = (edges_x[i,j,k]) or (edges_y[i,j,k]) or (edges_z[i,j,k])
    
    
    return edges

def accumulate_gradients(r_table, grayImage):
    '''
    Perform a General Hough Transform with the given image and R-table
    '''
    
    #Choose Edge Detector as desired
    edges = canny_edges_3d(grayImage) 
    #edges = sobel_edges_3d(grayImage)
    
    phi, psi = gradient_orientation(edges)
    
    accumulator = np.zeros(grayImage.shape)

    #print("Start Accumulation")
    for (i,j,k),value in np.ndenumerate(edges):

        if value:
            #Changed to int(gradient) which makes more sense
            for r in r_table[(int(phi[i,j,k]), int(psi[i,j,k]))]:
                accum_i, accum_j, accum_k = i+r[0], j+r[1], k+r[2]
                if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1] and accum_k < accumulator.shape[2]:
                    accumulator[int(accum_i), int(accum_j), int(accum_k)] += 1
                    
    return accumulator

def general_hough_closure(reference_image):
    '''
    Generator function to create a closure with the reference image and origin
    at the center of the reference image
    
    Returns a function f, which takes a query image and returns the accumulator
    '''
    
    referencePoint = (reference_image.shape[0]/2, reference_image.shape[1]/2, reference_image.shape[2]/2)
    
    print("Reference Point: ", referencePoint)
    
    r_table = build_r_table(reference_image, referencePoint)
    
    def f(query_image):
        return accumulate_gradients(r_table, query_image)
        
    return f

def n_max(a, n):
    '''
    Return the N max elements and indices in a
    '''
    indices = (-a.ravel()).argsort()[:n]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]
 
def test_general_hough(gh, reference_image, query):
    '''
    Uses a GH closure to detect shapes in an image and create nice output
    '''
    #query_image = imread(query, flatten=True)
    query_image = query
    query_dim = np.shape(query)
    reference_dim = np.shape(reference_image)
    
    accumulator = gh(query_image)
    
    #Filter accumulator
    kernel = np.array([[[0.1, 0.6, 0.1], [0.6, 0.8, 0.6], [0.1, 0.6, 0.1]],[[0.6, 0.8, 0.6], [0.8, 1.0, 0.8], [0.6, 0.8, 0.6]],[[0.1, 0.6, 0.1], [0.6, 0.8, 0.6], [0.1, 0.6, 0.1]]])
    signal.convolve(accumulator,kernel,mode="same")
    
    return accumulator

def test():

    ac_num = "9049401"
    
    sample_acs = []
    
    for file_name in os.listdir("no_fractures"):
        if file_name.find("sample.pkl") != -1:
            if file_name.find(str(ac_num)) == -1:
                sample_acs.append(file_name)
    
    print("ACCESION NUMBER: ", ac_num)
    
    #Open Downsized Pickle File Containing DICOM Scan
    dicom_downsized = cPickle.load(open("no_fractures/dicom_3d_" + ac_num + "_dwn4x.pkl","rb"),encoding = 'latin1')
    print("Size of Downsized Dicom Input: ", np.shape(dicom_downsized))
    
    dicom_downsized_dim = np.shape(dicom_downsized)
    
    accumulator = np.zeros(dicom_downsized_dim)
    
    for sample_ac in sample_acs:
        print("SAMPLE AC: ", sample_ac)
        
        #Open up the Sample that is used as the reference image
        c12_vertebrae = cPickle.load(open("no_fractures/" + sample_ac,"rb"),encoding = 'latin1')
        print("Size of Reference Detection Image: ", np.shape(c12_vertebrae))
    
    
    
        detect_s = general_hough_closure(c12_vertebrae)
    
        #Append onto the accumulator matrix the values acquired by each sample used
        accumulator = accumulator + test_general_hough(detect_s, c12_vertebrae, dicom_downsized)
    
    
    fig = plt.figure()
    plt.gray()
    
    #fig.add_subplot(2,2,1)
    #plt.title('Reference image')
    #plt.imshow(reference_image[:,:,reference_dim[2]//2])

    fig.add_subplot(2,2,1)
    plt.title('Query image')
    plt.imshow(dicom_downsized[:,:,dicom_downsized_dim[2]//2])
    
    
    fig.add_subplot(2,2,2)
    plt.title('Accumulator')
    plt.imshow(accumulator[:,:,dicom_downsized_dim[2]//2])
    
    fig.add_subplot(2,2,3)
    plt.title('Detection of Top 30 Points')
    plt.imshow(dicom_downsized[:,:,dicom_downsized_dim[2]//2])

#Get numerous top results that can be filtered out
    m = n_max(accumulator, 50)

    points = []
    x_pts = [] 
    y_pts = []
    z_pts = []
    
    highest_prob = m[0][0]
    
    for pt in m:
        #Filter out results so that only the most likely ones get chosen
        if pt[0] > 0.5*highest_prob:
            points.append((pt[1][0],pt[1][1],pt[1][2], pt[0]))
        
            x_pts.append(pt[1][0])
            y_pts.append(pt[1][1]) 
            z_pts.append(pt[1][2])
    
    plt.scatter(y_pts,x_pts, marker='o', color='r')
        
    print ("Top 30 Most Likely Points (x,y,z,certainty): ", points)



    fig.add_subplot(2,2,4)
    plt.title('Detection of Top 10 Points')
    plt.imshow(dicom_downsized[:,:,dicom_downsized_dim[2]//2])

    plt.show()

    #Get numerous top results that can be filtered out
    top10 = n_max(accumulator, 10)

    top10_points = []
    top10_x_pts = [] 
    top10_y_pts = []
    top10_z_pts = []
    
    highest_prob = top10[0][0]
    
    for pt in top10:
        #Filter out results so that only the most likely ones get chosen
        if pt[0] > 0.5*highest_prob:
            top10_points.append((pt[1][0],pt[1][1],pt[1][2], pt[0]))
        
            top10_x_pts.append(pt[1][0])
            top10_y_pts.append(pt[1][1]) 
            top10_z_pts.append(pt[1][2])    
    plt.scatter(top10_y_pts,top10_x_pts, marker='o', color='g')

    plt.show()
    
'''
    #Testing with 3D Images

    #Testing with a hollow cube
    dicom_3d = np.zeros((40,40,40))
    #Make both YZ planes have white sides
    dicom_3d[10:15,10:30,10:30] = 127
    dicom_3d[25:30,10:30,10:30] = 127
    
    #XZ Planes
    dicom_3d[10:30,10:15,10:30] = 127
    dicom_3d[10:30,25:30,10:30] = 127
    
    #XY Planes
    dicom_3d[10:30,10:30,10:15] = 127
    dicom_3d[10:30,10:30,25:30] = 127

    print(np.shape(dicom_3d))
    #print(dicom_3d[3,3,4])
    
    
    #Creating a test image
    test_3d = np.zeros((40,40,80))
    test_3d[0:40,0:40,25:45] = dicom_3d[0:40,0:40,10:30]
    test_3d[13:27,13:27,28:42] = 0
    test_3d[2:22,10:30,2:22] = 170
    test_3d[6:18,14:26,6:18] = 0
    
    
    #fig = plt.figure()
    #fig.add_subplot(1,2,1)
    #plt.imshow(dicom_3d[:,:,20])
    #fig.add_subplot(1,2,2)
    #plt.imshow(test_3d[:,:,20])
    
    #plt.show()
'''

    

if __name__ == '__main__':
    os.chdir("C:\\Users\\yoons\\Documents\\4th Year Semester 1\\ESC499 - Thesis\\Undergraduate_Thesis_Scripts\\DicomSubsampling")
        
    #os.chdir('C:\\Users\\yoons\\Documents\\4th Year Semester 1\\ESC499 - Thesis\\Undergraduate Thesis Scripts')
    test()
    
    