
from pylab import *

def logistic(x):
    return 1.0 / (1 + math.exp(-x))

x = [xx/10.0 for xx in range(-100, 100)]
y = [logistic(xx) for xx in x]

figure()
plot(x,y)
title('logistic function')
savefig('logistic_function.png')