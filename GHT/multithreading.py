import time
import threading

def calc_square(numbers):
    print("square numbers")
    for n in numbers:
        time.sleep(0.1)
        print('square:',n*n)
        

def calc_cube(numbers):
    print("cube numbers")
    for n in numbers:
        time.sleep(0.5)
        print('cube:',n*n*n)
        

def nested_forloop(n):
    print("nested for loop")
    counter = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                counter = counter + 1
             
def nested_forloop2(m):
    print("nested for loop")
    counter2 = 0
    for a in range(m):
        for b in range(m):
            for c in range(m):
                counter = counter2 + 1
                


arr1 = [1,3,5,7]
arr2 = [2,4,6,9]
        
t = time.time()
'''
calc_cube(arr1)
calc_cube(arr2)
calc_square(arr1)
calc_square(arr2)
'''

'''
t1 = threading.Thread(target = calc_cube, args=(arr1,))
t2 = threading.Thread(target = calc_cube, args=(arr2,))
t3 = threading.Thread(target = calc_square, args=(arr1,))
t4 = threading.Thread(target = calc_square, args=(arr2,))

t1.start()
t2.start()
t3.start()
t4.start()

t1.join()
t2.join()
t3.join()
t4.join()
'''


nested_forloop(200)
nested_forloop2(200)

print("done in : ", time.time() - t)
t = time.time()


t1 = threading.Thread(target=nested_forloop, args=(200,))
t2 = threading.Thread(target=nested_forloop2, args=(200,))
t1.start()
t2.start()
#t3.start()
t1.join()
t2.join()
#t3.join()


print("done in : ", time.time() - t)
