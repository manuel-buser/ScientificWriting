# src/__main__.py

from pathlib import Path
from src.download_data import main as download_data_main
from src.data import make_loader
from src.model_dense import DenseEncoder
from src.model_moe import MoEEncoder
from src.utils import measure_inference

import torch
from transformers import AutoTokenizer
import yaml


def run_test_pipeline(model_type="dense", cfg_path="conf/dense.yaml"):
    """
    Initializes and runs a minimal test of the full data → model → inference pipeline.
    """

    # --- Step 1: Ensure dataset exists ---
    data_file = Path("data/abstracts.jsonl")
    if not data_file.exists():
        print("📥 No data found — downloading sample abstracts...")
        download_data_main()

    # --- Step 2: Load config ---
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Step 3: Initialize data loader ---
    print("🧠 Loading dataset...")
    loader = make_loader(path="data/abstracts.jsonl",
                         batch_size=cfg.get("batch_size", 4),
                         seq_len=cfg.get("seq_len", 128))

    # --- Step 4: Initialize tokenizer and model ---
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    vocab_size = tok.vocab_size

    print(f"🚀 Initializing {model_type.upper()} model...")
    if model_type == "dense":
        model = DenseEncoder(vocab_size=vocab_size)
    elif model_type == "moe":
        model = MoEEncoder(vocab_size=vocab_size,
                           n_experts=cfg.get("n_experts", 4),
                           moe_layer_index=cfg.get("moe_layer_index", 1))
    else:
        raise ValueError("Unknown model type. Use 'dense' or 'moe'.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # --- Step 5: Run quick inference test ---
    print("⚙️ Running sample inference batch...")
    for i, batch in enumerate(loader):
        m = measure_inference(model, batch)
        print(f"Batch {i}: {m}")
        if i == 1:
            break

    print("✅ Test pipeline completed successfully.")


if __name__ == "__main__":
    # Run small integration test
    run_test_pipeline("dense", "conf/dense.yaml")
