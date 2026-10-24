import os,base64,hashlib,html,math,random
from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
import torch
from model import TinyTransformer,generate

app=FastAPI(title='NXT AI — own-model backend')
origins=[s.strip() for s in os.getenv('ALLOWED_ORIGINS','http://localhost:5500').split(',') if s.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type'])
model=None
checkpoint=Path(os.getenv('MODEL_PATH','checkpoint.pt'))
if checkpoint.exists():
    model=TinyTransformer()
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model']);model.eval()

class Request(BaseModel):
    prompt:str=Field(min_length=1,max_length=2000)

def encode(svg):return base64.b64encode(svg.encode()).decode()
def palette(prompt):
    seed=int(hashlib.sha256(prompt.encode()).hexdigest()[:12],16)
    rng=random.Random(seed)
    return rng,[f'hsl({rng.randrange(360)},75%,{rng.randrange(45,75)}%)' for _ in range(5)]

@app.get('/health')
def health():return {'ok':True,'model_loaded':model is not None,'image_engine':'procedural','video_engine':'animated-svg'}

@app.post('/api/chat')
def chat(req:Request):
    if model is None:raise HTTPException(503,'No trained model checkpoint loaded. Train and deploy checkpoint.pt first.')
    return {'answer':generate(model,req.prompt)[len(req.prompt):]}

@app.post('/api/image')
def image(req:Request):
    rng,colors=palette(req.prompt)
    circles=''.join(f'<circle cx="{rng.randint(0,900)}" cy="{rng.randint(0,600)}" r="{rng.randint(25,180)}" fill="{rng.choice(colors)}" opacity=".45"/>' for _ in range(24))
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="600" viewBox="0 0 900 600"><defs><linearGradient id="bg"><stop stop-color="#101329"/><stop offset="1" stop-color="{colors[0]}"/></linearGradient><filter id="blur"><feGaussianBlur stdDeviation="35"/></filter></defs><rect width="900" height="600" fill="url(#bg)"/><g filter="url(#blur)">{circles}</g><text x="42" y="550" fill="white" font-family="sans-serif" font-size="22">NXT ART / {html.escape(req.prompt[:48])}</text></svg>'
    return {'image_base64':encode(svg),'kind':'procedural-svg'}

@app.post('/api/video')
def video(req:Request):
    rng,colors=palette(req.prompt)
    circles=''.join(f'<circle cx="{rng.randint(80,720)}" cy="{rng.randint(60,390)}" r="{rng.randint(15,65)}" fill="{c}" opacity=".8"><animate attributeName="cx" values="{rng.randint(80,250)};{rng.randint(550,720)};{rng.randint(80,250)}" dur="{rng.randint(3,8)}s" repeatCount="indefinite"/></circle>' for c in colors)
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450"><rect width="800" height="450" fill="#0b1023"/>{circles}<text x="35" y="410" font-family="sans-serif" font-size="20" fill="white">NXT MOTION / {html.escape(req.prompt[:40])}</text></svg>'
    return {'animation_base64':encode(svg),'kind':'procedural-animated-svg'}
