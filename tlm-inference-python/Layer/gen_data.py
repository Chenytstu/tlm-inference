import sys
sys.path.append("/root/TLM-inference/tlm-inference-python")

import numpy as np
from Configs.model import *
from Linear.beaver import beaver
from NonLinear.NonLinear import gen_mask

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
    write(np.random.uniform(-.1, .1, (batch_size, inp_seq_length)), "./Layer/input.dat")
    
    size = [inp_seq_length, inp_seq_length, inp_seq_length, d_module, batch_size]
    bv = []
    for i in size:
        bv.append(beaver(i))
    write((bv[0][0], bv[0][1], bv[0][2], bv[1][0], bv[1][1], bv[1][2], bv[2][0], bv[2][1], bv[2][2], bv[3][0], bv[3][1], bv[3][2], bv[4][0], bv[4][1], bv[4][2]),  "./Layer/data/Attention_beaver1.dat")
    write((bv[0][3], bv[0][4], bv[0][5], bv[1][3], bv[1][4], bv[1][5], bv[2][3], bv[2][4], bv[2][5], bv[3][3], bv[3][4], bv[3][5], bv[4][3], bv[4][4], bv[4][5]),  "./Layer/data/Attention_beaver2.dat")
    mask = gen_mask(2, 2)
    write([[mask[0][0]], [mask[1][0]]], "./Layer/data/Attention_mask1.dat")
    write([[mask[0][1]], [mask[1][1]]], "./Layer/data/Attention_mask2.dat")
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WQ1.dat")
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WQ2.dat")
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WK1.dat")
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WK2.dat")
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WV1.dat")
    write(np.random.uniform(-.1, .1, (inp_seq_length, d_module)), "./Layer/data/Attention_WV2.dat")
    
    size = [d_module, d_module, d_module]
    bv = []
    for i in size:
        bv.append(beaver(i))
    write((bv[0][0], bv[0][1], bv[0][2], bv[1][0], bv[1][1], bv[1][2], bv[2][0], bv[2][1], bv[2][2]),  "./Layer/data/FFN_beaver1.dat")
    write((bv[0][3], bv[0][4], bv[0][5], bv[1][3], bv[1][4], bv[1][5], bv[2][3], bv[2][4], bv[2][5]),  "./Layer/data/FFN_beaver2.dat")
    mask = gen_mask(2, 2)
    write([[mask[0][0]], [mask[1][0]]], "./Layer/data/FFN_mask1.dat")
    write([[mask[0][1]], [mask[1][1]]], "./Layer/data/FFN_mask2.dat")
    write(np.random.uniform(-.1, .1, (d_module, d_module)), "./Layer/data/FFN_W11.dat")
    write(np.random.uniform(-.1, .1, (d_module, d_module)), "./Layer/data/FFN_W12.dat")
    write(np.random.uniform(-.1, .1, (d_module, d_module)), "./Layer/data/FFN_W21.dat")
    write(np.random.uniform(-.1, .1, (d_module, d_module)), "./Layer/data/FFN_W22.dat")
    write(np.random.uniform(-.1, .1, (1, d_module)), "./Layer/data/FFN_b11.dat")
    write(np.random.uniform(-.1, .1, (1, d_module)), "./Layer/data/FFN_b12.dat")
    write(np.random.uniform(-.1, .1, (1, d_module)), "./Layer/data/FFN_b21.dat")
    write(np.random.uniform(-.1, .1, (1, d_module)), "./Layer/data/FFN_b22.dat")
    
    size = [1]
    bv = []
    for i in size:
        bv.append(beaver(i))
    write((bv[0][0], bv[0][1], bv[0][2]),  "./Layer/data/LayerNorm_beaver1.dat")
    write((bv[0][3], bv[0][4], bv[0][5]),  "./Layer/data/LayerNorm_beaver2.dat")