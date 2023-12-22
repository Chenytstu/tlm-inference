import sys
sys.path.append("/root/TLM-inference/tlm-inference-python")

import numpy as np
from Configs.model import *

def write(mat, path):
    buffer = ""
    for i in mat:
        buf = ""
        for j in i:
            buf = buf + str(j) + ' '
        buf = buf + '\n'
        buffer = buffer + buf
    with open(path, 'w', encoding="utf-8") as f:
        f.write(buffer)
        

if __name__ == "__main__":
    # write(np.random.uniform(-1, 1, (d_module, d_module)), "./Layer/data/FFN_W11")
    # write(np.random.uniform(-1, 1, (d_module, d_module)), "./Layer/data/FFN_W12")
    # write(np.random.uniform(-1, 1, (d_module, d_module)), "./Layer/data/FFN_W21")
    # write(np.random.uniform(-1, 1, (d_module, d_module)), "./Layer/data/FFN_W22")
    write(np.random.uniform(-1, 1, (1, d_module)), "./Layer/data/FFN_b11")
    write(np.random.uniform(-1, 1, (1, d_module)), "./Layer/data/FFN_b12")
    write(np.random.uniform(-1, 1, (1, d_module)), "./Layer/data/FFN_b21")
    write(np.random.uniform(-1, 1, (1, d_module)), "./Layer/data/FFN_b22")
    