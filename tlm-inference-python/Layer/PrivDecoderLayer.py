from PrivLayer import *
from PrivAttention import Attention
from PrivFFN import FFN
from PrivLayerNorm import LayerNorm

class DecoderLayer(Layer):
    def __init__(self, party, inp) -> None:
        super().__init__(party, inp)
        self.ln1 = None
        self.attn = None
        self.ln2 = None
        self.ffn = None
        
    def forward(self):
        self.ln1 = LayerNorm(self.party, self.inp)
        _forward = self.ln1.forward()
        self.attn = Attention(self.party, _forward)
        _forward = self.attn.forward()
        self.ln2 = LayerNorm(self.party, _forward)
        _forward = self.ln2.forward()
        self.ffn = FFN(self.party, _forward)
        _forward = self.ffn.forward()
        return _forward