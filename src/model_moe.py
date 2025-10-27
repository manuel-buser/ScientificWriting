import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_dense import TransformerBlock

class Expert(nn.Module):
    def __init__(self, d_model=256, d_ff=1024):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    def forward(self, x): return self.ff(x)

class Top2Router(nn.Module):
    def __init__(self, d_model=256, n_experts=4):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, x):
        # x: [B, T, D]
        logits = self.gate(x)             # [B, T, E]
        probs = F.softmax(logits, dim=-1)
        top2 = probs.topk(k=2, dim=-1)    # values, indices
        return top2.values, top2.indices  # [B,T,2], [B,T,2]

class MoEBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=1024, n_experts=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.router = Top2Router(d_model, n_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x, attn_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=(attn_mask==0) if attn_mask is not None else None)
        x = self.ln1(x + self.drop(a))
        # MoE FFN
        weights, idx = self.router(x)  # [B,T,2], [B,T,2]
        out = torch.zeros_like(x)
        for k in range(2):  # top-2
            chosen = idx[..., k]                # [B,T]
            w = weights[..., k].unsqueeze(-1)   # [B,T,1]
            # gather per token expert
            expert_out = torch.zeros_like(x)
            for e_id, expert in enumerate(self.experts):
                mask = (chosen == e_id).unsqueeze(-1)  # [B,T,1]
                if mask.any():
                    expert_out = expert_out + expert(x * mask)
            out = out + w * expert_out
        x = self.ln2(x + self.drop(out))
        return x

class MoEEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=3, n_heads=4, d_ff=1024, moe_layer_index=1, n_experts=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            if i == moe_layer_index:
                self.blocks.append(MoEBlock(d_model, n_heads, d_ff, n_experts))
            else:
                self.blocks.append(TransformerBlock(d_model, n_heads, d_ff))
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for b in self.blocks:
            x = b(x, attention_mask)
        return self.lm_head(x)
