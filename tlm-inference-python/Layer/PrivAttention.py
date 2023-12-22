import numpy as np

from PrivLayer import *
from Configs.constant import *

class Attention(Layer):
    def __init__(self, party, port_offset=0) -> None:
        super().__init__(party)
        self.WQ = load_mat("Layer/data/Attention_WQ" + str(party) + ".dat")
        self.WK = load_mat("Layer/data/Attention_WK" + str(party) + ".dat")
        self.WV = load_mat("Layer/data/Attention_WV" + str(party) + ".dat")
        self.a, self.b, self.c, self.rand1, self.rand2 = load_parm("Layer/data/Attention_parm" + str(party) + ".dat")
        self.port_offset = port_offset
        
    def forward(self, inp):
        d_model, _ = self.WQ.shape
        Q = self.mult.matrix_multipication(inp, self.WQ, self.a, self.b, self.c, self.port_offset)
        K = self.mult.matrix_multipication(inp, self.WK, self.a, self.b, self.c, self.port_offset)
        V = self.mult.matrix_multipication(inp, self.WV, self.a, self.b, self.c, self.port_offset)
        Q_K = self.mult.matrix_multipication(Q, K.transpose(), self.a, self.b, self.c, self.port_offset) / famefrac(np.sqrt(d_model))
        softmax = []
        for i in Q_K:
            softmax.append(self.nonLinear.softmax(i, self.rand1, self.rand2, self.port_offset))
        softmax = np.asarray(softmax, dtype=object)
        return self.mult.matrix_multipication(softmax, V, self.a, self.b, self.c, self.port_offset)
