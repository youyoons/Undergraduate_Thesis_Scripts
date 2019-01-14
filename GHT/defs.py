def f(x):
    counter = 0
    for i in range(x):
        for j in range(x):
            for k in range(x):
                counter = counter + 1
    return counter
