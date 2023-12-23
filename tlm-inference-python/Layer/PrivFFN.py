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
        self.beaver = load_mat("Layer/data/FFN_beaver" + str(party) + ".dat")
        self.rand1, self.rand2 = load_parm("Layer/data/FFN_mask" + str(party) + ".dat")
        self.port_offset = port_offset
        
    def _forward_single(self, x):
        x1 = (self.mult.matrix_multipication(np.asarray([x], dtype=object), self.W1, self.beaver[0], self.beaver[1], self.beaver[2], self.port_offset)+ self.b1)[0]
        gelu = self.nonLinear.gelu(x1, self.beaver[3], self.beaver[4], self.beaver[5], self.rand1, self.rand2, self.port_offset)
        return (self.mult.matrix_multipication(np.asarray([gelu], dtype=object), self.W2, self.beaver[6], self.beaver[7], self.beaver[8], self.port_offset)+ self.b2)[0]
    
    def forward(self, inp):
        return np.asarray([self._forward_single(i) for i in inp], dtype=object)