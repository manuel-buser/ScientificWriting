from datasets import load_dataset
from pathlib import Path
import json

OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    # 1k abstracts from arXiv CS
    ds = load_dataset("arxiv_dataset", "cs", split="train[:1000]")
    # Keep only 'abstract'
    texts = [x["abstract"] for x in ds if x.get("abstract")]
    (OUT / "abstracts.jsonl").write_text(
        "\n".join(json.dumps({"text": t}) for t in texts), encoding="utf-8"
    )
    print(f"Saved {len(texts)} abstracts to data/abstracts.jsonl")

if __name__ == "__main__":
    main()
