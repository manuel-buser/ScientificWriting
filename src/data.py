from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class JsonlDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer_name="bert-base-uncased", seq_len=128):
        self.items = [json.loads(l) for l in Path(jsonl_path).read_text(encoding="utf-8").splitlines()]
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.seq_len = seq_len

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        text = self.items[idx]["text"]
        enc = self.tok(text, truncation=True, max_length=self.seq_len, padding="max_length", return_tensors="pt")
        # Simple next-token-style input/output on same sequence (demo)
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0)}

def make_loader(path="data/abstracts.jsonl", batch_size=8, seq_len=128):
    ds = JsonlDataset(path, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
