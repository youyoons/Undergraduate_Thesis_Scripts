import openpyxl
import os
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\yoons\\Documents\\4th Year Semester 2\\ESC499 - Thesis\\Undergraduate_Thesis_Scripts\\GHT")

book = openpyxl.load_workbook("../GHT/ground_truth_detection_pts_all.xlsx")
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
    
#print(ground_truth)
    
    
#Plot non-maximal suppression points
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

plt.savefig("ground_truth_distribution.png")
plt.show()

print(x_total/counter,y_total/counter,z_total/counter)