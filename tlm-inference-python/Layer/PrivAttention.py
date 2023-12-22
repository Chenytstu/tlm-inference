import numpy as np

from PrivLayer import *
from Configs.constant import *

class Attention(Layer):
    def __init__(self, party, inp) -> None:
        super().__init__(party, inp)
        self.WQ = load_mat("Layer/data/Attention_WQ" + str(party))
        self.WK = load_mat("Layer/data/Attention_WK" + str(party))
        self.WV = load_mat("Layer/data/Attention_WV" + str(party))
        self.a, self.b, self.c, self.rand1, self.rand2 = load_parm("Layer/data/Attention_parm" + str(party))
        
    def forward(self):
        d_model, _ = self.WQ.shape
        Q = self.mult.matrix_multipication(self.inp, self.WQ, self.a, self.b, self.c)
        K = self.mult.matrix_multipication(self.inp, self.WK, self.a, self.b, self.c)
        V = self.mult.matrix_multipication(self.inp, self.WV, self.a, self.b, self.c)
        Q_K = self.mult.matrix_multipication(Q, K.transpose(), self.a, self.b, self.c) / famefrac(np.sqrt(d_model))
        softmax = []
        for i in Q_K:
            softmax.append(self.nonLinear.softmax(i, self.rand1, self.rand2))
        softmax = np.asarray(softmax, dtype=object)
        return self.mult.matrix_multipication(softmax, V, self.a, self.b, self.c)