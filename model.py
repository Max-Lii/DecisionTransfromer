from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F 
from torch import Tensor
#
@dataclass
class ModelConfig:
    n_embd:int
    n_head:int
    state_dim:int
    act_dim:int
    n_layer:int
    dropout:float
    bias:bool
    vocab_size:int
    context_size:int
    gpt_wpe_wte_off:bool = False

#Casual attention
class CasualAttention(nn.Module):
    def __init__(self,config:ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # linear projection for getting Query, Key and Value 
        self.qkv_linear = nn.Linear(self.n_embd,3*self.n_embd,bias=config.bias)
        # linear layer after the concat 
        self.concat_linear = nn.Linear(self.n_embd,self.n_embd,bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self,x:Tensor,attn_mask:Tensor):
        #batch size, token length, embedding dimension(n_embd)
        B,T,C = x.size();
        q: torch.Tensor
        #mapping to input to query,key,value, split it at the last dim
        q,k,v = self.qkv_linear(x).split(self.n_embd,2);
        q = q.view(B,T,C // self.n_head, self.n_head).transpose(1,2); # (B,T,C)->(B,T,C_h,h)->(B,C_h,T,h)
        k = k.view(B,T,C // self.n_head, self.n_head).transpose(1,2); # (B,T,C)->(B,T,C_h,h)->(B,C_h,T,h)
        v = v.view(B,T,C // self.n_head, self.n_head).transpose(1,2); # (B,T,C)->(B,T,C_h,h)->(B,C_h,T,h)

        # using the flash attention automatically if possible
        # this function could not turn dropout off when in eval mode so we handle it ourselves
        attn_out = F.scaled_dot_product_attention(
            q,k,v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0 ,
            is_causal=True)
        
        out = self.resid_dropout(self.concat_linear(attn_out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self,config:ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.n_embd,4 * config.n_embd)
        self.linear_2 = nn.Linear(4 * config.n_embd,config.n_embd)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    def forward(self,x):
        x = self.linear_1(x)
        x = self.self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self,config:ModelConfig):
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.n_embd,bias=config.bias)
        self.attention = CasualAttention(config)
        self.ln_ffn = nn.LayerNorm(config.n_embd,bias=config.bias)
        self.ffn = FeedForward(config)

    def forward(self,x,attn_mask:Tensor):
        #According to GPT they use pre-norm instead which is different from original transformer 
        x = x + self.attention(self.ln_attn(x),attn_mask=attn_mask)
        x = x + self.ln_ffn(self.ln_ffn(x))
        return x
    
class GPT(nn.Module):
    """
        This is a modified version of GPT-2 specifically adjust for Decision transformer,
        when being use by Decision Transformer there will be no word embedding and positional encoding will be implementation in outside,
        so we provide a optional config to turn off word embedding and positional encoding.
    """
    def __init__(self,config:ModelConfig):
        super().__init__()
        #optional for turning off the wpe and wte
        self.gpt_wpe_wte_off = config.gpt_wpe_wte_off
        self.wpe = nn.Embedding(config.context_size, config.n_embd) if not config.gpt_wpe_wte_off else None
        self.wte = nn.Embedding(config.vocab_size,   config.n_embd) if not config.gpt_wpe_wte_off else None

        self.blocks = nn.ModuleList([ TransformerBlock for _ in config.n_layer ])
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.n_embd,bias=config.bias)

    def forward(self,x,attn_mask):
        if not self.gpt_wte_off:
            b, t = x.size()
            device = x.device
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            tok_emb = self.wte(x)
            pos_emb = self.wpe(pos)
            x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return x