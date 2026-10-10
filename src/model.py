import torch
from torch import nn
class GPT(nn.Module):
 def __init__(self,vocab,n_layer=2,d_model=128,n_head=4,context=128,dropout=.1):
  super().__init__(); self.context=context; self.emb=nn.Embedding(vocab,d_model); self.pos=nn.Embedding(context,d_model)
  layer=nn.TransformerEncoderLayer(d_model,n_head,4*d_model,dropout,batch_first=True,norm_first=True)
  self.blocks=nn.TransformerEncoder(layer,n_layer,enable_nested_tensor=False); self.norm=nn.LayerNorm(d_model); self.head=nn.Linear(d_model,vocab,bias=False)
 def forward(self,x):
  n=x.size(1); mask=torch.triu(torch.ones(n,n,device=x.device,dtype=torch.bool),1)
  return self.head(self.norm(self.blocks(self.emb(x)+self.pos(torch.arange(n,device=x.device))[None],mask=mask)))
