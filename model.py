from dataclasses import dataclass
from typing import Tuple
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F 
from torch import Tensor
from huggingface_hub import PyTorchModelHubMixin
#
@dataclass
class ModelConfig():
    n_embd:int
    n_head:int
    state_dim:int
    state_mean:np.ndarray
    state_std:np.ndarray
    act_dim:int
    n_layer:int
    dropout:float
    vocab_size:int
    context_size:int
    ep_len:int
    dt_bias:bool = True
    gpt_bias:bool = False
    action_tanh:bool = False
    gpt_wpe_wte_off:bool = False

#Casual attention
class CasualAttention(nn.Module):
    def __init__(self,config:ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # linear projection for getting Query, Key and Value 
        self.qkv_linear = nn.Linear(self.n_embd,3*self.n_embd,bias=config.gpt_bias)
        # linear layer after the concat 
        self.concat_linear = nn.Linear(self.n_embd,self.n_embd,bias=config.gpt_bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self,x:Tensor):
        #batch size, token length, embedding dimension(n_embd)
        B,T,C = x.size()
        q: torch.Tensor
        #mapping to input to query,key,value, split it at the last dim
        q, k, v = self.qkv_linear(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2); # (B,T,C)->(B,T,h,C_h)->(B,h,T,C_h)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2); # (B,T,C)->(B,T,h,C_h)->(B,h,T,C_h)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2); # (B,T,C)->(B,T,h,C_h)->(B,h,T,C_h)

        # using the flash attention automatically if possible
        # this function could not turn dropout off when in eval mode so we handle it ourselves

        # since we use right padding so we don't need to combine the mask now
        # if need to use left padding mask implementation of att_mask and padding_mask is needed
        attn_out = F.scaled_dot_product_attention(
            q,k,v,
            dropout_p=self.dropout if self.training else 0.0 ,
            is_causal=True)
        # re-assemble all head outputs side by side
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)# (B,h,T,C_h) ->(B,T,C)
        out = self.resid_dropout(self.concat_linear(attn_out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self,config:ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.n_embd,4 * config.n_embd,bias=config.gpt_bias)
        self.linear_2 = nn.Linear(4 * config.n_embd,config.n_embd,bias=config.gpt_bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    def forward(self,x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self,config:ModelConfig):
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.n_embd,bias=config.gpt_bias)
        self.attention = CasualAttention(config)
        self.ln_ffn = nn.LayerNorm(config.n_embd,bias=config.gpt_bias)
        self.ffn = FeedForward(config)

    def forward(self,x):
        #According to GPT they use pre-norm instead which is different from original transformer 
        x = x + self.attention(self.ln_attn(x))
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
        self.wpe = nn.Embedding(config.context_size, config.n_embd) if not config.gpt_wpe_wte_off else nn.Identity()
        self.wte = nn.Embedding(config.vocab_size,   config.n_embd) if not config.gpt_wpe_wte_off else nn.Identity()

        self.blocks = nn.ModuleList([ TransformerBlock(config) for _ in range(config.n_layer) ])
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.n_embd,bias=config.gpt_bias)

    def forward(self,x):
        if not self.gpt_wpe_wte_off:
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
class DecisionTransformer(nn.Module,PyTorchModelHubMixin):
    """
        Implementation for Decision Transformer
    """
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.context_size = config.context_size
        self.state_dim = config.state_dim
        self.act_dim = config.act_dim

        #normalization for state
        self.register_buffer("states_mean",torch.from_numpy(config.state_mean).view(1,1,self.state_dim))
        self.register_buffer("states_std",torch.from_numpy(config.state_std).view(1,1,self.state_dim))

        #embedding for position(time step), state, actions, returns-to-go
        self.pe = nn.Embedding(config.ep_len + config.context_size,self.n_embd) # extra time step embedding for padding 
        self.se = nn.Linear(config.state_dim,self.n_embd,bias=config.dt_bias)
        self.ae = nn.Linear(config.act_dim,self.n_embd,bias=config.dt_bias) 
        self.re = nn.Linear(1,self.n_embd,bias=config.dt_bias)

        #prediction head for action, state, return-to-go 
        self.pred_act = nn.Sequential(
            nn.Linear(self.n_embd,config.act_dim,bias=config.dt_bias),
            nn.Tanh() if config.action_tanh else nn.Identity(),
        )
        self.pred_state = nn.Linear(self.n_embd,config.state_dim,bias=config.dt_bias)
        self.pred_rtg= nn.Linear(self.n_embd,1,bias=config.dt_bias)
        self.embed_ln = nn.LayerNorm(self.n_embd)

        self.gpt = GPT(config)
    def forward(self,returns_to_go:Tensor, states:Tensor, actions:Tensor, time_steps, pad_mask = None)->Tuple[Tensor,Tensor,Tensor]:
        B,T = returns_to_go.shape[:2]
        device = states.device

        #normalization for state
        states = (states - self.states_mean) / self.states_std
        
        #embedding for r,s,a,time
        rtg_emb = self.re(returns_to_go)
        state_emb = self.se(states)
        act_emb = self.ae(actions)

       
        #position embedding for time steps
        # assert len(time_steps) == B,f"The element numbers:{len(time_steps)} in time steps array should meet the length of bath size:{B}"
        pos = torch.stack([torch.arange(max(t-self.context_size,0),t,dtype=torch.int) for t in time_steps],dim=0).to(device) # shape (t))
        pos_emb = self.pe(pos)

        #adding position embedding through broadcasting mechanism for better efficiency
        B,T,C = rtg_emb.size()
        
        traj = torch.stack((rtg_emb,state_emb,act_emb),dim=2) + pos_emb.unsqueeze(2) # (3,B,T,C) -> (B,T,3,C)
        traj = traj.view(B,3*T,C) # (B,3T,C): each time steps have 3 tokens
        traj = self.embed_ln(traj)
        # Notice: The official code do a layer norm here,
        # however, GPT using the pre-norm, which means token will apply layer norm before every blocks
        # so we don't have to use the layer norm here

        out:Tensor =  self.gpt(traj)

        #reconstruct to separate the to returns-to-go, state, action
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        out = out.view(B,T,3,C).transpose(1,2) #(B,3T,C) -> (B,T,3,C) -> (B,3,T,C)

        # predict next return given state and action 
        # use the a(t) to predict s'(t+1)
        # so when calculate the loss the predict S' should shift one time steps to align with the label 
        # and s'(0) will not be predict 
        rtg_pred = self.pred_rtg(out[:,2]) 

        # predict next state given state and action
        # use the a(t) to predict r'(t+1)   
        state_pred = self.pred_state(out[:,2]) 

        # predict next action given state
        # use the s(t) to predict a'(t)
        act_pred = self.pred_act(out[:,1])    

        # The original code of DT has a attention mask 
        # but since we use right padding 
        # we only need to add the mask before output

        if pad_mask is not None:
            rtg_pred = rtg_pred * pad_mask.unsqueeze(-1)
            state_pred = state_pred * pad_mask.unsqueeze(-1).repeat(1,1,self.state_dim)
            act_pred = act_pred * pad_mask.unsqueeze(-1).repeat(1,1,self.act_dim)

        return rtg_pred, state_pred, act_pred
    
    @torch.inference_mode()
    def get_action(self,returns_to_go, states, actions, time_steps,max_length=None):
                
        states = states[:,-self.context_size:]
        actions = actions[:,-self.context_size:]
        returns_to_go = returns_to_go[:,-self.context_size:]
        # time_steps = [ min(time_step,self.context_size)  for time_step in time_steps]
        
        rtg_pred, state_pred, act_pred = self.forward(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            time_steps=time_steps,
        )
        return act_pred[:,-1]