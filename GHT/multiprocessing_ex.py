import os
from multiprocessing import Pool
import defs
import time

os.chdir("C:\\Users\\Yoonsun You\\Documents")

t = time.time()

def g(x):
    counter = 0
    for i in range(x):
        for j in range(x):
            for k in range(x):
                counter = counter + 1
    return counter

if __name__ == '__main__':
    '''
    with Pool(10) as p:

        results = p.map(defs.f,[300,320,340,360,380,400,420,440,460,480])
        print(results)
    
    print(time.time() - t)
    '''

    t = time.time()
    
    a = g(300)
    b = g(320)
    c = g(340)
    d = g(360)
    e = g(380)
    f = g(400)
    k = g(420)
    
    h = g(440)
    i = g(460)
    j = g(480)
    
    print(time.time() - t)
    
