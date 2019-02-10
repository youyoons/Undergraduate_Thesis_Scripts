import os
import matplotlib.pyplot as plt
import numpy as np
import math


os.chdir("C:\\Users\\yoons\\Documents\\4th Year Semester 2\\Undergraduate_Thesis_Scripts\\GHT")
#os.chdir("C:\\Users\\yoons\\Documents\\ESC499\\Undergraduate_Thesis_Scripts\\GHT")

x1 = 0
x2 = 58
y1 = 17
y2 = 85

prior = np.zeros((x2-x1,y2-y1))

#Using Prior Distribution (about average centre of Ground Truth Points)
#Centre Variables
x_c = (x2+x1)//2
y_c = (y2+y1)//2

#Width
x_w = (x2-x1)//2
y_w = (y2-y1)//2
pwr = 4

#Just use widths as we are starting from the hard cut-out region
for dim1 in range(x2-x1):
    for dim2 in range(y2-y1):
        if (float(dim1-x_w)/x_w)**pwr + (float(dim2 - y_w)/y_w)**pwr <= 1:
            prior[dim1][dim2] = math.pow(1 - (float(dim1 - x_w)/x_w)**pwr - (float(dim2 - y_w)/y_w)**pwr,math.pow(pwr,-1))
      
            
plt.gray()
plt.imshow(prior)
plt.show()

print(prior[0:20,4:10])