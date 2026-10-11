import torch
from torch import nn
class GPT(nn.Module):
 def __init__(self,vocab,n_layer=2,d_model=128,n_head=4,context=128,dropout=.1):
  super().__init__(); self.context=context; self.emb=nn.Embedding(vocab,d_model); self.pos=nn.Embedding(context,d_model)
