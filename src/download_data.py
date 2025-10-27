from datasets import load_dataset
from pathlib import Path
import json

OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    OUT = Path("data")
    OUT.mkdir(parents=True, exist_ok=True)
    data_file = OUT / "abstracts.jsonl"

    if data_file.exists():
        print("📂 Found existing dataset — skipping download.")
        return

    print("📥 No data found — downloading abstracts from arXiv (CS category)...")
    ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train[:1000]")
    texts = [x["abstract"] for x in ds if x.get("abstract")]
    data_file.write_text(
        "\n".join(json.dumps({"text": t}) for t in texts), encoding="utf-8"
    )
    print(f"✅ Saved {len(texts)} abstracts to {data_file}")



if __name__ == "__main__":
    main()
