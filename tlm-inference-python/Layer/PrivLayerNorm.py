import numpy as np

from PrivLayer import *
from NonLinear.NonLinear import NonLinear

class LayerNorm(Layer):
    def __init__(self, party, gamma=1, beta=0, port_offset=0) -> None:
        super().__init__(party)
        self.gamma = gamma
        self.beta = beta
        self.beaver = load_parm("Layer/data/LayerNorm_beaver" + str(party) + ".dat")
        self.port_offset = port_offset
        
    def forward(self, inp):
        return np.asarray([self.nonLinear.layerNorm(
            i, 
            self.beaver[0], 
            self.beaver[1], 
            self.beaver[2], 
            self.gamma, 
            self.beta, 
            self.port_offset) for i in inp], 
                          dtype=object)
