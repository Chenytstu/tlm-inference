import numpy as np

def beaver(len: int):
    a1 = np.random.uniform(-1, 1, (len,))
    a2 = np.random.uniform(-1, 1, (len,))
    b1 = np.random.uniform(-1, 1, (len,))            
    b2 = np.random.uniform(-1, 1, (len,))
    c1 = np.random.uniform(-1, 1, (len,))
    c2 = (a1 + a2) * (b1 + b2) - c1
    return a1, b1, c1, a2, b2, c2