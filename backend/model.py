import torch
from torch import nn

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=256, width=128, heads=4, layers=2, context=128):
        super().__init__()
        self.context=context
        self.embedding=nn.Embedding(vocab_size,width)
        self.position=nn.Embedding(context,width)
        block=nn.TransformerEncoderLayer(d_model=width,nhead=heads,dim_feedforward=width*4,batch_first=True,dropout=0.0)
        self.blocks=nn.TransformerEncoder(block,num_layers=layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(width)
        self.head=nn.Linear(width,vocab_size)
    def forward(self,tokens):
        b,t=tokens.shape
        if t>self.context: raise ValueError('Context length exceeded')
        positions=torch.arange(t,device=tokens.device)
        x=self.embedding(tokens)+self.position(positions)[None,:,:]
        mask=torch.triu(torch.ones(t,t,device=tokens.device,dtype=torch.bool),diagonal=1)
        return self.head(self.norm(self.blocks(x,mask=mask,is_causal=True)))

@torch.no_grad()
def generate(model,prompt,max_tokens=120,temperature=0.9):
    model.eval()
    tokens=torch.tensor([list(prompt.encode('utf-8'))[-model.context:]],dtype=torch.long)
    for _ in range(max_tokens):
        logits=model(tokens[:,-model.context:])[:,-1,:]/max(temperature,0.1)
        next_token=torch.multinomial(torch.softmax(logits,dim=-1),1)
        tokens=torch.cat((tokens,next_token),dim=1)
    return bytes(tokens[0].tolist()).decode('utf-8',errors='replace')
