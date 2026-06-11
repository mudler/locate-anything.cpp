#!/usr/bin/env python3
"""Download nvidia/LocateAnything-3B into a local cache dir (default ./models/LocateAnything-3B).

Usage: python scripts/download_model.py [--out models/LocateAnything-3B] [--repo nvidia/LocateAnything-3B]
"""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nvidia/LocateAnything-3B")
    ap.add_argument("--out", default="models/LocateAnything-3B")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(out),
        allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "*.py",
            "merges.txt", "vocab.json", "added_tokens.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "chat_template.json", "preprocessor_config.json",
        ],
    )
    print(f"downloaded to {path}")

if __name__ == "__main__":
    main()
