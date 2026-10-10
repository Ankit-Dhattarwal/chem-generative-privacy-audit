import re
TOKEN_RE = re.compile(r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|\*|\$|%[0-9]{2}|[0-9])')
SPECIAL = ['<pad>', '<bos>', '<eos>']
def tokenize(s): return TOKEN_RE.findall(s)
class Tokenizer:
 def __init__(self, smiles):
  self.itos = SPECIAL + sorted({t for s in smiles for t in tokenize(s)})
  self.stoi = {t:i for i,t in enumerate(self.itos)}
  self.pad_id,self.bos_id,self.eos_id = (self.stoi[x] for x in SPECIAL)
 def encode(self,s): return [self.bos_id]+[self.stoi[t] for t in tokenize(s)]+[self.eos_id]
