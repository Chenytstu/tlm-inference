import numpy as np

from PrivLayer import *
from NonLinear.NonLinear import NonLinear

class LayerNorm(Layer):
    def __init__(self, party, input, gamma=1, beta=0) -> None:
        super().__init__(party, input)
        self.gamma = gamma
        self.beta = beta
        self.a, self.b, self.c = load_parm("Layer/data/LayerNorm_parm" + str(party))
        
    def forward(self):
        return np.asarray([self.nonLinear.layerNorm(i, self.a, self.b, self.c, self.gamma, self.beta) for i in self.inp], 
                          dtype=object)
