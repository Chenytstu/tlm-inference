import sys
sys.path.append("/root/TLM-inference/tlm-inference-python")

import numpy as np

from Layer.PrivDecoderLayer import DecoderLayer
from Configs.constant import *
from Configs.model import *

if __name__ == "__main__":
    party = int(sys.argv[1])
    inp = []
    for i in np.random.uniform(-1, 1, (input_size, batch_size)):
        tmp = []
        for j in i:
            tmp.append(famefrac(j))
        inp.append(tmp)
    inp = np.asarray(inp, dtype=object)
    import time
    
    layer = DecoderLayer(party, inp)
    start = time.time()
    _forward = layer.forward()
    end = time.time() - start
    