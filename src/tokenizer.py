import re
TOKEN_RE = re.compile(r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|\*|\$|%[0-9]{2}|[0-9])')
SPECIAL = ['<pad>', '<bos>', '<eos>']
def tokenize(s): return TOKEN_RE.findall(s)
class Tokenizer:
 def __init__(self, smiles):
  self.itos = SPECIAL + sorted({t for s in smiles for t in tokenize(s)})
