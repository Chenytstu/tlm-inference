import sys
sys.path.append("/root/TLM-inference/tlm-inference-python")

from Configs.constant import *
from Linear.Mult import Mult
from NonLinear.NonLinear import NonLinear
from role import Alice, Bob

import numpy as np

def load_mat(path: str):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        for i in f.readlines():
            tmp = []
            for j in i[:-2].split(" "):
                tmp.append(famefrac(float(j)))
            data.append(tmp)
    return np.asarray(data, dtype=object)
                
def load_parm(path: str):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        for i in f.readlines():
            data.append(famefrac(float(i)))
    return np.asarray(data, dtype=object)

class Layer:
    def __init__(self, party, inp) -> None:
        assert party == Alice or party == Bob
        self.party = party
        self.inp = inp
        self.mult = Mult(party)
        self.nonLinear = NonLinear(party)
    
    def forward(self):
        pass