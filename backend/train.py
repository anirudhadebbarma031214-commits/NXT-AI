import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from model import TinyTransformer

p=argparse.ArgumentParser()
p.add_argument('--data',default='data.txt')
p.add_argument('--steps',type=int,default=300)
p.add_argument('--out',default='checkpoint.pt')
args=p.parse_args()
torch.manual_seed(42)
raw=Path(args.data).read_bytes()
context=128
if len(raw)<context+2: raise SystemExit('Training file must contain at least 130 bytes')
data=torch.tensor(list(raw),dtype=torch.long)
model=TinyTransformer(context=context)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
model.train()
for step in range(args.steps):
    offsets=torch.randint(0,len(data)-context-1,(16,))
    x=torch.stack([data[i:i+context] for i in offsets])
    y=torch.stack([data[i+1:i+context+1] for i in offsets])
    loss=F.cross_entropy(model(x).reshape(-1,256),y.reshape(-1))
    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
    if step%50==0:print(f'step {step}: loss {loss.item():.3f}',flush=True)
torch.save({'model':model.state_dict(),'architecture':{'context':context}},args.out)
print('Saved checkpoint:',args.out)
