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
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WQ1.dat")
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WQ2.dat")
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WK1.dat")
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WK2.dat")
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WV1.dat")
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WV2.dat")
    write(np.random.uniform(-.1, .1, (d_module, d_module)), "./Layer/data/FFN_W11.dat")
    write(np.random.uniform(-.1, .1, (d_module, d_module)), "./Layer/data/FFN_W12.dat")
    write(np.random.uniform(-.1, .1, (d_module, d_module)), "./Layer/data/FFN_W21.dat")
    write(np.random.uniform(-.1, .1, (d_module, d_module)), "./Layer/data/FFN_W22.dat")
    write(np.random.uniform(-.1, .1, (1, d_module)), "./Layer/data/FFN_b11.dat")
    write(np.random.uniform(-.1, .1, (1, d_module)), "./Layer/data/FFN_b12.dat")
    write(np.random.uniform(-.1, .1, (1, d_module)), "./Layer/data/FFN_b21.dat")
    write(np.random.uniform(-.1, .1, (1, d_module)), "./Layer/data/FFN_b22.dat")
    write(np.random.uniform(-.1, .1, (batch_size, inp_seq_length)), "./Layer/input.dat")
    