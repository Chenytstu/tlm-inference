import sys
sys.path.append("/root/TLM-inference/tlm-inference-python")

from Layer.PrivLayer import *
from Layer.PrivDecoderLayer import DecoderLayer
from Configs.constant import *
from Configs.model import *

if __name__ == "__main__":
    party = int(sys.argv[1])
    inp = load_mat("Layer/input.dat")
    
    import time
    start = time.time()
    layer = DecoderLayer(party)
    end = time.time() - start
    print("parm loaded, time:", end)
    _forward = layer.forward(inp)
    print(inp_seq_length)
    # show(_forward)
    