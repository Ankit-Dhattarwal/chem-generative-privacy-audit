import os, re, random, csv
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from .tokenizer import tokenize

def load_dataset(path='data/moses.csv', pool_size=2000, val_size=500, seed=0):
 # Reservoir-sample source rows to keep preprocessing within laptop memory.
 rng=random.Random(seed); reservoir=[]; seen=0
 with open(path,newline='') as fh:
  reader=csv.DictReader(fh)
  col='SMILES' if 'SMILES' in (reader.fieldnames or []) else (reader.fieldnames or [''])[0]
  for row in reader:
   s=row.get(col)
   if not s: continue
   seen+=1
   if len(reservoir)<100000: reservoir.append(s)
   else:
    j=rng.randrange(seen)
    if j<100000: reservoir[j]=s
 uniq={}
 for s in reservoir:
  m=Chem.MolFromSmiles(s)
  if m is None: continue
  c=Chem.MolToSmiles(m,canonical=True)
  if len(tokenize(c))<=100: uniq[c]=m
 vals=sorted(uniq); rng=np.random.default_rng(seed); ix=rng.permutation(len(vals))
 if len(ix)<pool_size+val_size: raise ValueError('dataset too small after filtering')
 val=[vals[i] for i in ix[:val_size]]; pool=[vals[i] for i in ix[val_size:val_size+pool_size]]
 os.makedirs('data',exist_ok=True)
 pd.DataFrame({'id':range(len(pool)),'smiles':pool}).to_csv('data/pool.csv',index=False)
 pd.DataFrame({'id':range(len(val)),'smiles':val}).to_csv('data/val.csv',index=False)
 sc=[MurckoScaffold.MurckoScaffoldSmiles(mol=uniq[s]) for s in pool]; freq=pd.Series(sc).value_counts().to_dict()
 rows=[]
 for i,s in enumerate(pool):
  mol=uniq[s]; variants={s}
  for _ in range(30):
   if len(variants)>=10: break
   variants.add(Chem.MolToSmiles(mol,doRandom=True,canonical=False))
  rows.append({'id':i,'smiles':s,'scaffold':sc[i],'scaffold_frequency':freq[sc[i]],'variants':'|'.join(sorted(variants)[:10])})
 pd.DataFrame(rows).to_csv('data/pool_metadata.csv',index=False)
 assert not set(pool)&set(val)
 return pool,val
