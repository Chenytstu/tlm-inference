import numpy as np

from Communication.api import *
from Configs.constant import *
from role import *
import Configs.communication as config

class Mult:
    def __init__(self, party: int):
        assert(party == Alice or party == Bob)
        self._party = party
        
    def hadamard_product(self, x, y, a, b, c, port_offset=0):
        s = x - a
        t = y - b
        s_remote = None
        t_remote = None
        if self._party == Alice:
            send((s, t), port=config.default_port_1 + port_offset)
        else: 
            s_remote, t_remote = recv(port=config.default_port_1 + port_offset)
            s += s_remote
            t += t_remote
            s_remote = s
            t_remote = t
            send((s, t), port=config.default_port_2 + port_offset)
            return s_remote * b + t_remote * a + c
        s_remote, t_remote = recv(port=config.default_port_2 + port_offset)
        s = s_remote
        t = t_remote
        return s_remote * b + t_remote * a + c + s * t
    
    def vector_multipication(self, x, y, a, b, c, port_offset=0):
        return self.hadamard_product(x, y, a, b, c, port_offset)
    
    def matrix_multipication(self, x, y, a, b, c, port_offset=0):
        s = x.copy()
        t = y.transpose().copy()
        for i in range(len(s)):
            s[i] -= a
        for j in range(len(t)):
            t[j] -= b
        s_remote = None
        t_remote = None
        if self._party == Alice:
            send((s, t), port=config.default_port_1 + port_offset)
        else:
            s_remote, t_remote = recv(port=config.default_port_1 + port_offset)
            s += s_remote
            t += t_remote
            s_remote = s
            t_remote = t
            send((s, t), port=config.default_port_2 + port_offset)
            dim1, _ = s.shape
            dim2, _ = t.shape
            z = []
            for i in range(dim1):
                tmp = []
                for j in range(dim2):
                    # z[i][j] = s[i] * b + t[j] * a + c * _
                    tmp.append(famcfrac(sum(s[i] * b) + sum(t[j] * a) + c * _))
                z.append(tmp)
            return np.asarray(z, dtype=object)
        s_remote, t_remote = recv(port=config.default_port_2 + port_offset)
        s = s_remote
        t = t_remote
        dim1, _ = s.shape
        dim2, _ = t.shape
        z = []
        for i in range(dim1):
            tmp = []
            for j in range(dim2):
                # z[i][j] = s[i].dot(b) + t[j].dot(a) + c * _ + s[i].dot(t[j])
                tmp.append(famcfrac(sum(s[i] * b) + sum(t[j] * a) + c * _ + s[i].dot(t[j])))
            z.append(tmp)
        return np.asarray(z, dtype=object)
        
def show(x):
    for i in x:
        for j in i:
            print(j, end=" ")
        print()

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
            tmp.append(famcfrac(np.random.uniform(-1, 1)))
        y.append(tmp)
    a = [1, 1]
    b = [1, 1]
    c = 2
    x, y, a, b = np.asarray(x, dtype=object), np.asarray(y, dtype=object), np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    party = int(sys.argv[1])
    mul = Mult(party)
    
    z = mul.matrix_multipication(x, y, a, b, c)
    if party == Alice:
        send((x, y, z))
    else:
        x_remote, y_remote, z_remote = recv()
        show(z + z_remote - (x + x_remote).dot(y + y_remote))
    