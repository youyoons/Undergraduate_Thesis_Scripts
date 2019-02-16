import openpyxl
import os
import matplotlib.pyplot as plt
import numpy as np
import math

#os.chdir("C:\\Users\\yoons\\Documents\\4th Year Semester 2\\ESC499 - Thesis\\Undergraduate_Thesis_Scripts\\GHT")
os.chdir("C:\\Users\\yoons\\Documents\\ESC499\\Undergraduate_Thesis_Scripts\\GHT")

book = openpyxl.load_workbook("../GHT/ground_truth_detection_pts_validation_set.xlsx")
sheet = book.active
row_count = sheet.max_row

ground_truth = {}

x_total = 0
y_total = 0
z_total = 0
counter = 0

for i in range(3,row_count): #divided by 3 as a test 
    
    ac_num_loc = sheet.cell(row = i,column = 1)
    ac_num = str(ac_num_loc.value)
    
    x = sheet.cell(row = i, column = 2).value
    y = sheet.cell(row = i, column = 3).value
    z = sheet.cell(row = i, column = 4).value

    
    if (x != None) and (y != None) and (z != None):
        ground_truth[ac_num] = [x,y,z]
    
        x_total = x_total + x
        y_total = y_total + y
        z_total = z_total + z
        counter = counter + 1

    
#Plot non-maximal suppression points
#*******************************************************************************
nms_x_pts = [] 
nms_y_pts = []

for key in ground_truth.keys():
    nms_x_pts.append(ground_truth[key][0])
    nms_y_pts.append(ground_truth[key][1])  

plt.title("Ground Truth Detection Point Distribution")
plt.ylim(128,0)
plt.xlim(0,128)
plt.xlabel('y')
plt.ylabel('x')

plt.scatter(nms_y_pts,nms_x_pts, marker='o', color='g')

plt.scatter(y_total/counter, x_total/counter, marker= 'X', color = 'r')


#Plot the Prior Distribution Boundary
#*******************************************************************************
#Power of ellipse-shape
pwr = 6

#Right half of prior distribution limit
ellipse_x1 = np.zeros(59)
ellipse_y1 = np.zeros(59)

for i in range(59):
    ellipse_x1[i] = i
    ellipse_y1[i] = math.pow(34**pwr - (34*(i-29)/29)**pwr, math.pow(pwr,-1)) + 51
    
plt.plot(ellipse_y1, ellipse_x1, marker = 'X', color = 'b')

#Left half of prior distribution limit
ellipse_x2 = np.zeros(59)
ellipse_y2 = np.zeros(59)

for i in range(59):
    ellipse_x2[i] = i
    ellipse_y2[i] = -math.pow(34**pwr - (34*(i-29)/29)**pwr, math.pow(pwr,-1)) + 51
    
plt.plot(ellipse_y2, ellipse_x2, marker = 'X', color = 'b')


#Plot the hard bounding box
#*******************************************************************************
bound_sq_x = [0, 0, 58 , 58, 0]
bound_sq_y = [17, 85, 85, 17, 17]

plt.plot(bound_sq_y, bound_sq_x, marker = '.', color = 'k')


#Save Float and Show it
#*******************************************************************************
plt.savefig("ground_truth_distribution.png")
plt.show()

print(x_total/counter,y_total/counter,z_total/counter)