import numpy as np, torch
import os
from torch.nn.utils.rnn import pad_sequence
from .model import GPT

def train_model(smiles, tokenizer, val, epochs=30, seed=0, model_dir='runs/model', device='cpu', n_layer=2,d_model=128):
 os.makedirs(model_dir,exist_ok=True)
 torch.manual_seed(seed); np.random.seed(seed); dev=torch.device(device)
 model=GPT(len(tokenizer.itos),n_layer=n_layer,d_model=d_model).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
 def batch_data(ss): return pad_sequence([torch.tensor(tokenizer.encode(s)[:128]) for s in ss],batch_first=True,padding_value=tokenizer.pad_id)
 x=batch_data(smiles).to(dev); vx=batch_data(val).to(dev); logs=[]
 for ep in range(1,epochs+1):
  model.train(); order=torch.randperm(len(x),device=dev); total=0; count=0
  for ids in order.split(32):
   b=x[ids]; logits=model(b[:,:-1]); loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),b[:,1:].reshape(-1),ignore_index=tokenizer.pad_id)
   opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); total+=loss.item();count+=1
  model.eval()
  with torch.no_grad(): vl=torch.nn.functional.cross_entropy(model(vx[:,:-1]).reshape(-1,len(tokenizer.itos)),vx[:,1:].reshape(-1),ignore_index=tokenizer.pad_id).item()
  logs.append({'epoch':ep,'train_loss':total/count,'val_loss':vl})
  if ep in (5,20,30,60): torch.save({'model':model.state_dict(),'epoch':ep},f'{model_dir}/epoch{ep}.pt')
 return model,logs

def sequence_nll(model, tokenizer, smiles, device='cpu'):
 model.eval(); out=[]
 from torch.nn.utils.rnn import pad_sequence
 encoded=[tokenizer.encode(s)[:128] for s in smiles]
 with torch.no_grad():
  for start in range(0,len(encoded),128):
   x=pad_sequence([torch.tensor(z) for z in encoded[start:start+128]],batch_first=True,padding_value=tokenizer.pad_id).to(device)
   lp=torch.log_softmax(model(x[:,:-1]),-1).gather(-1,x[:,1:].unsqueeze(-1)).squeeze(-1)
   mask=x[:,1:]!=tokenizer.pad_id; losses=-lp*mask; sums=losses.sum(1); means=sums/mask.sum(1).clamp_min(1)
   for j in range(len(sums)):
    token_losses=losses[j,mask[j]].cpu().numpy()
    out.append((float(sums[j]),float(means[j]),token_losses))
 return out
