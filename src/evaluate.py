import torch, yaml, argparse
from transformers import AutoTokenizer
from src.data import make_loader
from src.model_dense import DenseEncoder
from src.model_moe import MoEEncoder
from src.utils import measure_inference, append_metrics

def build_model(cfg, vocab_size):
    if cfg["model"] == "dense":
        return DenseEncoder(vocab_size)
    elif cfg["model"] == "moe":
        return MoEEncoder(vocab_size,
                          n_experts=cfg.get("n_experts", 4),
                          moe_layer_index=cfg.get("moe_layer_index", 1))
    else:
        raise ValueError("Unknown model")

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    loader = make_loader(seq_len=cfg["seq_len"], batch_size=cfg["batch_size"])
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    model = build_model(cfg, vocab_size=tok.vocab_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # a few batches for timing stability
    for i, batch in enumerate(loader):
        m = measure_inference(model, batch)
        m.update({"model": cfg["model"], "batch": i, "device": device})
        append_metrics(m)
        if i == 9: break  # 10 batches

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    main(args.cfg)
