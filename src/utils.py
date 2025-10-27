import time, torch, pandas as pd
from pathlib import Path

def measure_inference(model, batch):
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    device = next(model.parameters()).device
    with torch.inference_mode():
        start = time.perf_counter()
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start
    mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    tokens = batch["input_ids"].numel()
    return dict(seconds=elapsed, tokens=tokens, sec_per_token=elapsed/tokens, peak_mem_bytes=mem)

def append_metrics(row: dict, path="results/metrics.csv"):
    p = Path(path); p.parent.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame([row])
    if p.exists(): df.to_csv(p, mode="a", header=False, index=False)
    else: df.to_csv(p, index=False)
