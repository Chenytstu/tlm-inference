from Configs.model import head
from PrivLayer import *
from PrivAttention import Attention
from PrivFFN import FFN
from PrivLayerNorm import LayerNorm

def show(x):
    for i in x:
        for j in i:
            print(j, end=" ")
        print()
class DecoderLayer(Layer):
    def __init__(self, party) -> None:
        super().__init__(party)
        step = 100
        # self.ln1 = LayerNorm(self.party, port_offset=-1 * step)
        # self.attn = [Attention(self.party, port_offset=i * step) for i in range(head)]
        # self.ln2 = LayerNorm(self.party, port_offset=head * step)
        # self.ffn = FFN(self.party, port_offset=(head + 1) * step)
        self.ln1 = LayerNorm(self.party, layer=1)
        self.attn = [Attention(self.party) for _ in range(head)]
        self.ln2 = LayerNorm(self.party, layer=1)
        self.ffn = FFN(self.party)
        
        
    def forward(self, inp):
        import time
        all_time = []
        
        start = time.time()
        _forward = self.ln1.forward(inp)
        all_time.append(time.time() - start)
        #print("layer norm 1 finished, time:", time.time() - start)
        
        start = time.time()
        _forward1 = self.attn[head-1].forward(_forward)
        all_time.append(time.time() - start)
        #print("attention 1 finished, time:", time.time() - start)
        for i in range(head - 1):
            start = time.time()
            _forward1 += self.attn[i].forward(_forward)
            all_time.append(time.time() - start)
            #print(f"attention {i + 2} finished, time:", time.time() - start)
        _forward = _forward1

        start = time.time()
        _forward = self.ln2.forward(_forward)
        all_time.append(time.time() - start)
        #print("layer norm 2 finished, time:", time.time() - start)
        
        start = time.time()
        _forward = self.ffn.forward(_forward)
        all_time.append(time.time() - start)
        #print("ffn finished, time:", time.time() - start)
        
        print("time cost:", sum(all_time))
        return _forward