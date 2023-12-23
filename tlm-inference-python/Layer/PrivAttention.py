import numpy as np

from PrivLayer import *
from Configs.constant import *

class Attention(Layer):
    def __init__(self, party, port_offset=0) -> None:
        super().__init__(party)
        self.WQ = load_mat("Layer/data/Attention_WQ" + str(party) + ".dat")
        self.WK = load_mat("Layer/data/Attention_WK" + str(party) + ".dat")
        self.WV = load_mat("Layer/data/Attention_WV" + str(party) + ".dat")
        self.beaver = load_mat("Layer/data/Attention_beaver" + str(party) + ".dat")
        self.rand1, self.rand2 = load_parm("Layer/data/Attention_mask" + str(party) + ".dat")
        self.port_offset = port_offset
        
    def forward(self, inp):
        d_model, _ = self.WQ.shape
        Q = self.mult.matrix_multipication(inp, self.WQ, self.beaver[0], self.beaver[1], self.beaver[2], self.port_offset)
        K = self.mult.matrix_multipication(inp, self.WK, self.beaver[3], self.beaver[4], self.beaver[5], self.port_offset)
        V = self.mult.matrix_multipication(inp, self.WV, self.beaver[6], self.beaver[7], self.beaver[8], self.port_offset)
        Q_K = self.mult.matrix_multipication(Q, K.transpose(), self.beaver[9], self.beaver[10], self.beaver[11], self.port_offset) / famefrac(np.sqrt(d_model))
        softmax = []
        for i in Q_K:
            softmax.append(self.nonLinear.softmax(i, self.rand1, self.rand2, self.port_offset))
        softmax = np.asarray(softmax, dtype=object)
        return self.mult.matrix_multipication(softmax, V, self.beaver[12], self.beaver[13], self.beaver[14], self.port_offset)
