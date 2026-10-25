"""Common-input author-architecture FT port with source-only epoch selection.

This is an architecture port, not the original paper's best-weight reproduction.
"""
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from .features import FEATURES,CATS,NUMS
DEFAULT_CONFIG={'width':32,'depth':2,'heads':4,'dropout':.1,'batch_size':64,'lr':.001,'weight_decay':.0001,'minimum_epochs':10,'max_epochs':150,'patience':20}

def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

class Encoder:
    """Train-only scaling and train-only categorical vocabulary with zero OOV."""
    def fit(self, data):
        self.mean = data[NUMS].to_numpy(float).mean(axis=0)
        spread = data[NUMS].to_numpy(float).std(axis=0)
        # Match StandardScaler's constant-column convention. A source constant
        # cannot define a 1e-8-sized measurement scale for an unseen value.
        self.sd = np.where(spread <= 1e-8, 1.0, spread)
        self.vocab = {c: {s: i+1 for i, s in enumerate(sorted(data[c].astype(str).unique()))} for c in CATS}
        return self

    def transform(self, data):
        num = ((data[NUMS].to_numpy(float)-self.mean)/self.sd).astype('float32')
        cat = np.column_stack([data[c].astype(str).map(self.vocab[c]).fillna(0).to_numpy(int) for c in CATS])
        return torch.from_numpy(num), torch.from_numpy(cat).long()

    def payload(self):
        return {'numeric_features': NUMS, 'categorical_features': CATS,
                'numeric_mean': self.mean.tolist(), 'numeric_sd': self.sd.tolist(),
                'vocabulary': self.vocab, 'oov_code': 0, 'oov_embedding': 'fixed zero',
                'constant_numeric_scale': 1.0, 'constant_threshold': 1e-8}

class AuthorAttention(nn.Module):
    """Keras MHA key_dim=d_model means each head has d_model dimensions."""
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.width, self.heads = width, heads
        self.qkv = nn.Linear(width, 3*heads*width)
        self.out = nn.Linear(heads*width, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.width).permute(2,0,3,1,4).unbind(0)
        weights = self.drop(torch.softmax((q @ k.transpose(-1,-2))/math.sqrt(self.width), dim=-1))
        z = (weights @ v).transpose(1,2).reshape(b,t,self.heads*self.width)
        return self.out(z)

class AuthorBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.att = AuthorAttention(width, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,width))
        self.norm2 = nn.LayerNorm(width, eps=1e-6)

    def forward(self, x):
        z = self.norm1(x)
        z = z + self.att(z)
        return self.norm2(z + self.ff(z))

class AuthorFTPort(nn.Module):
    def __init__(self, cardinalities, width=32, depth=2, heads=4, dropout=.1):
        super().__init__()
        self.numeric_w = nn.Parameter(torch.empty(len(NUMS),width))
        self.numeric_b = nn.Parameter(torch.empty(len(NUMS),1))
        self.emb = nn.ModuleList([nn.Embedding(n,width,padding_idx=0) for n in cardinalities])
        self.blocks = nn.ModuleList([AuthorBlock(width,heads,dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.readout = nn.Sequential(nn.Linear(width,16),nn.Linear(16,1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.numeric_w,std=.05)
        nn.init.normal_(self.numeric_b,std=.05)
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for e in self.emb:
            nn.init.uniform_(e.weight,-.05,.05)
            with torch.no_grad(): e.weight[0].zero_()

    def forward(self, numeric, categorical):
        num = torch.relu(numeric[:,:,None]*self.numeric_w[None,:,:]+self.numeric_b[None,:,:])
        cat = torch.stack([e(categorical[:,j]) for j,e in enumerate(self.emb)],dim=1)
        x = torch.cat([num,cat],dim=1)
        for block in self.blocks: x = block(x)
        # Public AI4Water default has no CLS token: read first numeric token.
        return self.readout(self.norm(x[:,0,:])).squeeze(-1)

def predict_log(model, encoder, frame, batch_size=256):
    model.eval(); num,cat=encoder.transform(frame); pp=[]
    with torch.no_grad():
        for start in range(0,len(frame),batch_size):
            pp.append(model(num[start:start+batch_size],cat[start:start+batch_size]).numpy())
    return np.concatenate(pp)

def fit_model(frame, config, seed, validation=None, fixed_epochs=None):
    seed_all(seed)
    encoder = Encoder().fit(frame)
    model = AuthorFTPort([len(encoder.vocab[c])+1 for c in CATS],
                         width=config['width'],depth=config['depth'],heads=config['heads'],dropout=config['dropout'])
    num,cat = encoder.transform(frame)
    y = torch.tensor(np.log(frame.qe.to_numpy()),dtype=torch.float32)
    with torch.no_grad(): model.readout[-1].bias.fill_(float(y.mean()))
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    best_loss=float('inf');best_epoch=0;best_state=None;history=[]
    total_epochs = fixed_epochs or config['max_epochs']
    for epoch in range(1,total_epochs+1):
        model.train(); order=torch.randperm(len(frame)); total_loss=0.
        for start in range(0,len(frame),config['batch_size']):
            ix=order[start:start+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss=torch.mean((model(num[ix],cat[ix])-y[ix])**2)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite FT training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();total_loss+=float(loss.detach())*len(ix)
        value={'epoch':epoch,'train_ln_mse':total_loss/len(frame)}
        if validation is not None:
            vp=predict_log(model,encoder,validation)
            val_loss=float(np.mean((vp-np.log(validation.qe.to_numpy()))**2))
            value['validation_ln_mse']=val_loss
            if val_loss < best_loss-1e-6:
                best_loss=val_loss;best_epoch=epoch
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            if epoch>=config['minimum_epochs'] and epoch-best_epoch>=config['patience']:
                history.append(value);break
        else:
            best_epoch=epoch
        history.append(value)
    if validation is not None: model.load_state_dict(best_state)
    return model,encoder,pd.DataFrame(history),best_epoch,best_loss
