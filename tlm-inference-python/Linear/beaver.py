import numpy as np

from FixedPoint import *

def beaver(len: int, hadamard_product = False):
    a1 = np.random.uniform(-1, 1, (len,))
    a2 = np.random.uniform(-1, 1, (len,))
    b1 = np.random.uniform(-1, 1, (len,))            
    b2 = np.random.uniform(-1, 1, (len,))
    c1 = np.random.uniform(-1, 1, (len,))
    c2 = (a1 + a2) * (b1 + b2) - c1
    if (hadamard_product):
        return a1, a2, b1, b2, c1.sum(), c2.sum()
    return a1, a2, b1, b2, c1, c2
