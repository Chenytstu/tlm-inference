import sys
sys.path.append("/root/tlm-inference-python")

import numpy as np
from Communication.api import *
from role import *
import Configs.communication as config

class NonLinear:
    def __init__(self, party: int):
        assert(party == Alice or party == Bob)
        self._party = party
    
    def inverse_square(self):
        pass
    
    def softmax(self):
        pass
    
    def gelu(self):
        pass
    