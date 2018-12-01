import os
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.misc import imread
from skimage.feature import canny
from scipy.ndimage.filters import sobel
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
from scipy import signal
from scipy.ndimage.filters import gaussian_filter
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

    return accumulator

'''
Function: GHT
Input: ac_num (an array of accession numbers)
Purpose: detects C1, C2 vertebrae and displays at both as points and plots
'''
def GHT(ac_num):
    #reference_acs = ["dicom_3d_8214092_reference.pkl"]
    reference_acs = []

    #Get the references that will be used
    for file_name in os.listdir("no_fractures"):
        if file_name.find("reference.pkl") != -1:
            if file_name.find(str(ac_num)) == -1:
                reference_acs.append(file_name)
    
    print("ACCESSION NUMBER: ", ac_num)
    
    
    #Open Downsized Pickle File Containing DICOM Scan
    dicom_dwn4x_pp = cPickle.load(open("no_fractures/dicom_3d_" + ac_num + "_dwn4x.pkl","rb"),encoding = 'latin1')
    dicom_dwn4x_pp_dim = np.shape(dicom_dwn4x_pp)
    print("Size of Downsized Dicom Input: ", dicom_dwn4x_pp_dim)

    #Specify Prior Limits to Volume
    x1 = 5
    x2 = 65
    y1 = 15
    y2 = 80

    
    #Get specific region of focus (based on prior information
    dicom_dwn4x = dicom_dwn4x_pp[x1:x2,y1:y2,:]
    dicom_dwn4x_dim = np.shape(dicom_dwn4x)
    print("Size of Specific Dicom: ", dicom_dwn4x_dim)


    #Initialize Accumulator that will be used to get top points
    accumulator = np.zeros(dicom_dwn4x_dim)
    
    for reference_ac in reference_acs:
        print("REFERENCE AC: ", reference_ac)
        
        #Open up the Reference that is used as the reference image
        reference = cPickle.load(open("no_fractures/" + reference_ac,"rb"),encoding = 'latin1')
        print("Size of Reference Image: ", np.shape(reference))
    
    
    
        detect_s = general_hough_closure(reference)
    
        #Get elementwise maximum of accumulator matrix when compared to new reference
        accumulator = np.maximum(accumulator,test_general_hough(detect_s, reference, dicom_dwn4x))
    
    final_accumulator = accumulator
    
    
    '''
    #Find reference that is most relevant to accumulator
    identical_elements = 0
    detected_refrence = []
    detected_reference_ac = ""
    FA = np.array(final_accumulator)
    print("The size of FA is: ", np.shape(FA))
    
    for reference_ac in reference_acs:
        print("REFERENCE AC: ", reference_ac)
        
        #Open up the Reference that is used as the reference image
        reference = cPickle.load(open("no_fractures/" + reference_ac,"rb"),encoding = 'latin1')
        print("Size of Reference Detection Image: ", np.shape(reference))
     
    
        detect_s = general_hough_closure(reference)
    
        #Get elementwise maximum of accumulator matrix when compared to new reference
        accumulator = test_general_hough(detect_s, reference, dicom_dwn4x)
        
        A = np.array(accumulator)
        print("The size of A is: ",np.shape(A))
        if np.sum(A==FA) > identical_elements:
            identical_elements = np.sum(A==FA)
            detected_reference_ac = reference_ac
            detected_reference = reference
            
    detected_reference_dim = np.shape(detected_reference)
    print(detected_reference_dim)
    print("The reference that detected the region of interest the best was: ", detected_reference_ac)
    '''
    
    std_dev = 1.0
    final_accumulator = gaussian_filter(final_accumulator,sigma = std_dev, order = 0)

    fig = plt.figure(num = ac_num + "_sigma" + str(std_dev), figsize = (12,12))
    plt.gray()

    fig.suptitle(ac_num + "_sigma" + str(std_dev))

    fig.add_subplot(2,2,1)
    plt.title('Query image')
    plt.imshow(dicom_dwn4x_pp[:,:,dicom_dwn4x_pp_dim[2]//2])
    
    
    fig.add_subplot(2,2,2)
    plt.title('Accumulator')
    plt.imshow(final_accumulator[:,:,dicom_dwn4x_dim[2]//2])
    
    fig.add_subplot(2,2,3)
    plt.title('Detection of Top 40 Points')
    plt.imshow(dicom_dwn4x_pp[:,:,dicom_dwn4x_dim[2]//2])

    #Get top 40 results that can be filtered out
    m = n_max(final_accumulator, 40)

    points = []
    x_pts = [] 
    y_pts = []
    z_pts = []
    
    #highest_prob = m[0][0]
    
    for pt in m:
        points.append((pt[1][0] + x1,pt[1][1] + y1,pt[1][2], int(pt[0])))
    
        x_pts.append(pt[1][0]+x1)
        y_pts.append(pt[1][1]+y1) 
        z_pts.append(pt[1][2])
    
    plt.scatter(y_pts,x_pts, marker='.', color='r')
        
    print ("Top 40 Most Likely Points (x,y,z,certainty): ", points)



    fig.add_subplot(2,2,4)
    plt.title('Points after Non-Maximal Suppression')
    plt.imshow(dicom_dwn4x_pp[:,:,dicom_dwn4x_pp_dim[2]//2])

    '''
    #Get numerous top results that can be filtered out
    top10 = n_max(final_accumulator, 10)

    top10_points = []
    top10_x_pts = [] 
    top10_y_pts = []
    top10_z_pts = []


    #highest_prob = top10[0][0]
    
    for pt in top10:
        top10_points.append((pt[1][0] + x1,pt[1][1] + y1,pt[1][2], pt[0]))
    
        top10_x_pts.append(pt[1][0]+x1)
        top10_y_pts.append(pt[1][1]+y1) 
        top10_z_pts.append(pt[1][2])    

    plt.scatter(top10_y_pts,top10_x_pts, marker='o', color='g')
    '''

    #Perform non-maximal suppression
    nms_pts = []
    
    for pt in points:
        if len(nms_pts) == 0:
            nms_pts.append(pt)
        else:
            counter = 0
            for i in range(len(nms_pts)):
                if math.sqrt((nms_pts[i][0]-pt[0])**2 + (nms_pts[i][1]-pt[1])**2 + (nms_pts[i][2]-pt[2])**2) > 10:
                    counter = counter + 1
                else:
                    if pt[3] > nms_pts[i][3]:
                        nms_pts[i] = pt
            
            if counter == len(nms_pts):
                nms_pts.append(pt)
  
        
    print("Non-Maximal Suppression Points: ", nms_pts)
  
  
    #Sliding reference across volume around detected points to find accurate point
    #reference_vol_pp = np.array(reference_sample)
    #reference_vol = np.ndarray.flatten(reference_vol_pp)
    #print(np.shape(reference_vol))
    
    optimal_pt = [0,0]
    max_cross_correl_val = -float('inf')
    
    for pt in nms_pts:
        print("The point being investigated is: ", pt)
        #print("The detected reference dim is: ", detected_reference_dim)
        for i in range(-8,9):
            for j in range(-8,9):
                for k in range(-2,3):
                    cross_correl_val = 0
                    for reference_ac in reference_acs:
                        reference_vol_pp1 = cPickle.load(open("no_fractures/" + reference_ac,"rb"),encoding = 'latin1')
                        reference_vol_pp2 = np.array(reference_vol_pp1)
                        reference_vol = np.ndarray.flatten(reference_vol_pp2)
                        reference_dim = np.shape(reference_vol_pp1)
                        
                        
                        x1 = pt[0] - reference_dim[0]//2 + i
                        x2 = x1 + reference_dim[0]
                        
                        y1 = pt[1] - reference_dim[1]//2 + j
                        y2 = y1 + reference_dim[1]
                        
                        z1 = pt[2] - reference_dim[2]//2 + k
                        z2 = z1 + reference_dim[2]
                        
                        #Exit current slide location if out of bounds
                        if x1 < 0 or y1 < 0 or z1 < 0:
                            break
                        
                        current_vol_pp = np.array(dicom_dwn4x_pp[x1:x2,y1:y2,z1:z2])
                        current_vol = np.ndarray.flatten(current_vol_pp)
                        
                        #print("Reference Dim: ",reference_dim)
                        #print("Current Dim: ",np.shape(current_vol_pp))
                        
                        if x1 < 0:
                            cross_correl_val = cross_correl_val + np.dot(reference_vol[reference_dim[0]-(x2-x1):reference_dim[0],:,:], current_vol)
                        else:
                            cross_correl_val = cross_correl_val + np.dot(reference_vol, current_vol)
                        
                    if cross_correl_val > max_cross_correl_val:
                        max_cross_correl_val = cross_correl_val
                        print("The cross correlation value is: ", cross_correl_val)
                        optimal_pt = [pt[0]+i,pt[1]+j, pt[2]+k]
                        print("The optimal point currently is: ", optimal_pt)
                    
    print("The Final Detection point is: ",optimal_pt)
    
    print(reference_acs)
                
                
    
  
    #Plot non-maximal suppression points
    nms_x_pts = [] 
    nms_y_pts = []
    nms_z_pts = []
    
    for pt in nms_pts:
        nms_x_pts.append(pt[0])
        nms_y_pts.append(pt[1]) 
        nms_z_pts.append(pt[2])    


    plt.scatter(nms_y_pts,nms_x_pts, marker='o', color='g')
    plt.scatter(optimal_pt[1],optimal_pt[0], marker='x', color='y')
    
    plt.show()
    
    #Save Figure
    #plt.savefig(ac_num + "_sigma" + str(std_dev) + ".png")

    

if __name__ == '__main__':
    os.chdir("C:\\Users\\yoons\\Documents\\4th Year Semester 1\\ESC499 - Thesis\\Undergraduate_Thesis_Scripts\\DicomSubsampling")
    
    ac_nums_pp = os.listdir("no_fractures/")
    ac_nums = []
    
    for ac_num_pp in ac_nums_pp:
        str1 = ac_num_pp.split("dicom_3d_")[1]
        str2 = str1.split("_")[0]
        
        ac_nums.append(str2)

    #Get the detection results for the given accession number(s)
    for ac_num in ac_nums[0:1]:
        print(ac_num)
        GHT(ac_num)
    
    