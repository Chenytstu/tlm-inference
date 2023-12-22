import numpy as np

from PrivLayer import *
from Configs.constant import *
from Linear.Mult import Mult
from NonLinear.NonLinear import NonLinear

class FFN(Layer):
    def __init__(self, party, inp) -> None:
        super().__init__(party, inp)
        self.W1 = load_mat("Layer/data/FFN_W1" + str(party))
        self.W2 = load_mat("Layer/data/FFN_W2" + str(party))
        self.b1 = load_mat("Layer/data/FFN_b1" + str(party))
        self.b2 = load_mat("Layer/data/FFN_b2" + str(party))
        self.a, self.b, self.c, self.rand1, self.rand2 = load_parm("Layer/data/FFN_parm" + str(party))
        
    def _forward_single(self, x):
        x1 = (self.mult.matrix_multipication(np.asarray([x], dtype=object), self.W1, self.a, self.b, self.c)+ self.b1)[0]
        gelu = self.nonLinear.gelu(x1, self.a, self.b, self.c, self.rand1, self.rand2)
        return (self.mult.matrix_multipication(np.asarray([gelu], dtype=object), self.W2, self.a, self.b, self.c)+ self.b2)[0]
    
    def forward(self):
        return np.asarray([self._forward_single(i) for i in self.inp], dtype=object)