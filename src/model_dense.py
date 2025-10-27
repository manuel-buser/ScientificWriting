import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        a, _ = self.attn(x, x, x, key_padding_mask=(attn_mask==0) if attn_mask is not None else None)
        x = self.ln1(x + self.drop(a))
        f = self.ff(x)
        x = self.ln2(x + self.drop(f))
        return x

class DenseEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=3, n_heads=4, d_ff=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for b in self.blocks:
            x = b(x, attention_mask)
        logits = self.lm_head(x)  # next-token style
        return logits
