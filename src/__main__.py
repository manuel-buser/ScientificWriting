# src/__main__.py
# ----------------
# Entry point for testing and benchmarking Dense vs. MoE Transformer models
# with adjustable hyperparameters and automatic result logging.

from pathlib import Path
from datetime import datetime
import torch
import yaml
import json
import argparse
from transformers import AutoTokenizer

from src.download_data import main as download_data_main
from src.data import make_loader
from src.model_dense import DenseEncoder
from src.model_moe import MoEEncoder
from src.utils import measure_inference

# -------------------------------------------------------------------------
# 🔧 Helper function: create unique output directory for each experiment
# -------------------------------------------------------------------------
def create_run_dir(base_dir="results"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# -------------------------------------------------------------------------
# 🚀 Core experiment pipeline
# -------------------------------------------------------------------------
def run_test_pipeline(model_type="dense", cfg_path="conf/dense.yaml", overrides=None):
    """
    Runs a full data → model → inference pipeline with flexible configuration.
    Logs results to a unique timestamped directory in `results/`.

    Args:
        model_type (str): 'dense' or 'moe'
        cfg_path (str): Path to YAML config
        overrides (dict): Optional dictionary of parameter overrides
    """

    # --- Step 1: Prepare data ------------------------------------------------
    data_file = Path("data/abstracts.jsonl")
    if not data_file.exists():
        print("📥 No data found — downloading sample abstracts...")
        download_data_main()

    # --- Step 2: Load base config --------------------------------------------
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply any runtime overrides (e.g., new d_model or n_layers)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})

    # --- Step 3: Create unique output folder ---------------------------------
    run_dir = create_run_dir()
    (run_dir / "config_used.yaml").write_text(yaml.dump(cfg), encoding="utf-8")

    # --- Step 4: Initialize data loader --------------------------------------
    print("🧠 Loading dataset...")
    loader = make_loader(
        path="data/abstracts.jsonl",
        batch_size=cfg.get("batch_size", 4),
        seq_len=cfg.get("seq_len", 128)
    )

    # --- Step 5: Initialize tokenizer and model ------------------------------
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    vocab_size = tok.vocab_size

    print(f"🚀 Initializing {model_type.upper()} model...")
    if model_type == "dense":
        model = DenseEncoder(
            vocab_size=vocab_size,
            d_model=cfg.get("d_model", 256),
            n_layers=cfg.get("n_layers", 3),
            n_heads=cfg.get("n_heads", 4),
            d_ff=cfg.get("d_ff", 1024)
        )
    elif model_type == "moe":
        model = MoEEncoder(
            vocab_size=vocab_size,
            d_model=cfg.get("d_model", 256),
            n_layers=cfg.get("n_layers", 3),
            n_heads=cfg.get("n_heads", 4),
            d_ff=cfg.get("d_ff", 1024),
            n_experts=cfg.get("n_experts", 4),
            moe_layer_index=cfg.get("moe_layer_index", 1)
        )
    else:
        raise ValueError("❌ Unknown model type. Use 'dense' or 'moe'.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # Save model summary
    model_summary = f"Model: {model_type}\nParams: {sum(p.numel() for p in model.parameters()):,}"
    (run_dir / "model_info.txt").write_text(model_summary, encoding="utf-8")

    # --- Step 6: Run timed inference -----------------------------------------
    print("⚙️ Running sample inference batches...")
    results = []
    for i, batch in enumerate(loader):
        metrics = measure_inference(model, batch)
        metrics.update({
            "batch": i,
            "model": model_type,
            "device": device,
            "cfg_path": cfg_path
        })
        print(f"Batch {i}: {metrics}")
        results.append(metrics)
        if i == 9:  # limit to 10 batches
            break

    # --- Step 7: Save results to file ----------------------------------------
    results_path = run_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"💾 Results saved to {results_path}")

    print("✅ Test pipeline completed successfully.")


# -------------------------------------------------------------------------
# 🧭 CLI Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run dense or MoE model benchmark.")
    parser.add_argument("--cfg", type=str, default="conf/dense.yaml", help="Path to YAML config")
    parser.add_argument("--model", type=str, choices=["dense", "moe"], default="dense", help="Model type")
    parser.add_argument("--d_model", type=int, default=None, help="Hidden dimension size")
    parser.add_argument("--n_layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=None, help="Feed-forward dimension")
    parser.add_argument("--n_experts", type=int, default=None, help="Number of experts (MoE only)")
    parser.add_argument("--seq_len", type=int, default=None, help="Sequence length override")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")

    args = parser.parse_args()

    # --- Load YAML config first ---
    import yaml
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Override parameters from CLI if provided ---
    for key in ["d_model", "n_layers", "n_heads", "d_ff", "n_experts", "seq_len", "batch_size"]:
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    # --- Save updated config temporarily for logging ---
    from datetime import datetime
    run_name = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    cfg["run_name"] = run_name

    print(f"🚀 Starting benchmark: {args.model.upper()} ({run_name})")
    print(f"Configuration:\n{cfg}")

    # --- Run pipeline ---
    run_test_pipeline(args.model, args.cfg, overrides={
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "n_experts": args.n_experts,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size
    })

