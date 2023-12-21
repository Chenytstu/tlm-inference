import sys
sys.path.append("/root/TLM-inference/tlm-inference-python")

import time
import numpy as np
from Communication.api import *
from role import *
import Configs.communication as config

class Mult:
    def __init__(self, party: int):
        assert(party == Alice or party == Bob)
        self._party = party
        
    def hadamard_product(self, x, y, a, b, c):
        s = x - a;
        t = y - b;
        s_remote = None
        t_remote = None
        if self._party == Alice:
            send((s, t), port=config.default_port_1)
        else: 
            s_remote, t_remote = recv(port=config.default_port_1)
            s += s_remote
            t += t_remote
            s_remote = s
            t_remote = t
            send((s, t), port=config.default_port_2)
            return s_remote * b + t_remote * a + c
        s_remote, t_remote = recv(port=config.default_port_2)
        s = s_remote
        t = t_remote
        return s_remote * b + t_remote * a + c + s * t
    
    def vector_multipication(self, x, y, a, b, c):
        return self.hadamard_product(x, y, a, b, c)
    
    def matrix_multipication(self, x, y, a, b, c):
        s = x.copy()
        t = y.transpose().copy()
        for i in range(len(s)):
            s[i] -= a
        for j in range(len(t)):
            t[j] -= b
        s_remote = None
        t_remote = None
        if self._party == Alice:
            send((s, t), port=config.default_port_1)
        else:
            s_remote, t_remote = recv(port=config.default_port_1)
            s += s_remote
            t += t_remote
            s_remote = s
            t_remote = t
            send((s, t), port=config.default_port_2,)
            dim1, _ = s.shape
            dim2, _ = t.shape
            z = np.zeros((dim1, dim2))
            for i in range(dim1):
                for j in range(dim2):
                    z[i][j] = s[i].dot(b) + t[j].dot(a) + c * _
            return z
        s_remote, t_remote = recv(port=config.default_port_2)
        s = s_remote
        t = t_remote
        dim1, _ = s.shape
        dim2, _ = t.shape
        z = np.zeros((dim1, dim2))
        for i in range(dim1):
            for j in range(dim2):
                z[i][j] = s[i].dot(b) + t[j].dot(a) + c * _ + s[i].dot(t[j])
        return z
        
    
if __name__ == "__main__":
    famcfrac = FXfamily(64)
    dim1, dim2, dim3 = 2, 2, 2
    x = []
    y = []
    for i in range(dim1):
        tmp = []
        for j in range(dim2):
            tmp.append(famcfrac(np.random.uniform(-1, 1)))
        x.append(tmp)
    for i in range(dim2):
        tmp = []
        for j in range(dim3):
            tmp.append(famcfrac(0))
        y.append(tmp)
    a = [famcfrac(1.) for _ in range(dim2)]
    b = [famcfrac(1.) for _ in range(dim2)]
    c = famcfrac(dim2 / 2.)
    x, y, a, b = np.asarray(x, dtype=object), np.asarray(y, dtype=object), np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    mul = Mult(int(sys.argv[1]))
    
    z = mul.matrix_multipication(x, y, a, b, c)
    