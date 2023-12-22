import numpy as np

from PrivLayer import *
from Configs.constant import *

class FFN(Layer):
    def __init__(self, party, port_offset=0) -> None:
        super().__init__(party)
        self.W1 = load_mat("Layer/data/FFN_W1" + str(party) + ".dat")
        self.W2 = load_mat("Layer/data/FFN_W2" + str(party) + ".dat")
        self.b1 = load_mat("Layer/data/FFN_b1" + str(party) + ".dat")
        self.b2 = load_mat("Layer/data/FFN_b2" + str(party) + ".dat")
        self.a, self.b, self.c, self.rand1, self.rand2 = load_parm("Layer/data/FFN_parm" + str(party) + ".dat")
        self.port_offset = port_offset
        
    def _forward_single(self, x):
        x1 = (self.mult.matrix_multipication(np.asarray([x], dtype=object), self.W1, self.a, self.b, self.c, self.port_offset)+ self.b1)[0]
        gelu = self.nonLinear.gelu(x1, self.a, self.b, self.c, self.rand1, self.rand2, self.port_offset)
        return (self.mult.matrix_multipication(np.asarray([gelu], dtype=object), self.W2, self.a, self.b, self.c, self.port_offset)+ self.b2)[0]
    
    def forward(self, inp):
        return np.asarray([self._forward_single(i) for i in inp], dtype=object)