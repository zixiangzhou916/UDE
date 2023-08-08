"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""
import os
import sys
import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

import importlib

class Block(nn.Module):
    def __init__(self, n_emb, n_head, block_size, attn_pdrop, resid_pdrop):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
        self.attn = CausalCrossConditionalSelfAttention(
            n_emb=n_emb, n_head=n_head, block_size=block_size, 
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(resid_pdrop),
        )
        
    def forward(self, x, cond_len):
        x = x + self.attn(self.ln1(x), cond_len=cond_len)
        x = x + self.mlp(self.ln2(x))
        return x
        
class CausalCrossConditionalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """
    def __init__(self, n_emb, n_head, block_size, attn_pdrop, resid_pdrop):
        super(CausalCrossConditionalSelfAttention, self).__init__()
        assert n_emb % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_emb, n_emb)
        self.query = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_emb, n_emb)
        
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        
        # output projection
        self.proj = nn.Linear(n_emb, n_emb)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        # self.mask = se
        self.n_head = n_head
        
    def forward(self, x, cond_len):
        """
        :param x: [batch_size, nframes, dim]
        """
        B, T, C = x.size()  # T = 3*t (music up down)
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        t = T - cond_len
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                
        mask = torch.zeros(1, 1, T, T).float().to(x.device)
        mask[:, :, :, :cond_len] = 1
        mask[:, :, -t:, -t:] = self.mask[:, :, :t, :t]
        att = att.masked_fill(mask == 0, float('-inf'))
        
        # ##### DEBUG
        # import numpy as np
        # mask = torch.tril(torch.ones(T, T), diagonal=cond_len).detach().cpu().numpy()
        # np.savetxt("mask.txt", mask, fmt="%d")
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class CrossCondGPTBase(nn.Module):
    """ the full GPT language model, with a context size of block_size """
    
    def __init__(self, conf):
        super(CrossCondGPTBase, self).__init__()
        
        self.tok_emb = nn.Embedding(num_embeddings=conf["n_tokens"], embedding_dim=conf["d_latent"])
        self.pos_emb = nn.Parameter(torch.zeros(1, conf["n_positions"], conf["d_latent"]), requires_grad=True)
        self.cond_emb = nn.Linear(conf["d_model"], conf["d_latent"])
        self.drop = nn.Dropout(conf["drop"])
        # transformer
        self.blocks = nn.ModuleList()
        for _ in range(conf["n_layers"]):
            self.blocks.append(
                Block(n_emb=conf["d_latent"], n_head=conf["n_head"], block_size=conf["block_size"], 
                      attn_pdrop=conf["attn_pdrop"], resid_pdrop=conf["resid_pdrop"])
            )
        
        self.block_size = conf["block_size"]
        self.apply(self._init_weights)
        
    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, idx, cond):
        
        b, t = idx.size()
        _, cond_t, _ = cond.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        # if self.requires_head:
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        token_embeddings = torch.cat([self.cond_emb(cond), token_embeddings], dim=1)

        pos_emb_1 = self.pos_emb[:, :cond_t, :]
        pos_emb_2 = self.pos_emb[:, cond_t:cond_t+t, :]
        position_embeddings = torch.cat([pos_emb_1, pos_emb_2], dim=1) # each position maps to a (learnable) vector
        
        x = self.drop(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x, cond_len=cond.shape[1])
        # x = self.ln_f(x)

        return x
            
class CrossCondGPTHead(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    
    def __init__(self, conf):
        super(CrossCondGPTHead, self).__init__()
        # transformer
        self.blocks = nn.ModuleList()
        for _ in range(conf["n_layers"]):
            self.blocks.append(
                Block(n_emb=conf["d_latent"], n_head=conf["n_head"], block_size=conf["block_size"], 
                      attn_pdrop=conf["attn_pdrop"], resid_pdrop=conf["resid_pdrop"])
            )

        # decoder head
        self.ln_f = nn.LayerNorm(conf["d_latent"])
        self.block_size = conf["block_size"]
        self.head = nn.Linear(conf["d_latent"], conf["n_tokens"]-1, bias=False) # 2048
        
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, x, target_length):
        
        _, T, _ = x.shape
        for block in self.blocks:
            x = block(x, cond_len=T-target_length)
        x = self.ln_f(x)
        N, T, C = x.size()
        logits = self.head(x[:, -target_length:])
        
        return logits
        
class CrossCondGPT(nn.Module):
    def __init__(self, conf):
        super(CrossCondGPT, self).__init__()
        self.base = importlib.import_module(
            conf["gpt_base"]["arch_path"], package="networks").__getattribute__(
                conf["gpt_base"]["arch_name"])(conf["gpt_base"])
        self.head = importlib.import_module(
            conf["gpt_head"]["arch_path"], package="networks").__getattribute__(
                conf["gpt_head"]["arch_name"])(conf["gpt_head"])     
            
    def forward(self, idx, cond):
        base_out = self.base(idx, cond)
        head_out = self.head(base_out, target_length=idx.size(1)) 
        return head_out
        
if __name__ == "__main__":
    
    import yaml
    
    with open("configs/vqgan/config_vqgan_ude_gpt_v1.yaml", "r") as f:
        conf = yaml.safe_load(f)
        
    model = CrossCondGPTBase(conf=conf["model"]["ude"]["gpt_base"])
    idx = torch.randint(0, 1024, size=(2, 30))
    cond = torch.randn(2, 100, 512).float()
    model(idx, cond)