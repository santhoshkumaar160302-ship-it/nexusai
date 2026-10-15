"""Ingest a directory of Markdown/text files into a running gateway over HTTP.

  python scripts/ingest_dir.py data/sample_docs --url http://localhost:8000 --api-key $NEXUS_API_KEY
"""
from __future__ import annotations

import argparse
from pathlib import Path

import httpx

SUFFIXES = {".md", ".markdown", ".txt", ".rst"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--api-key", default="")
    p.add_argument("--batch", type=int, default=20)
    args = p.parse_args()

    root = Path(args.directory)
    files = sorted(f for f in root.rglob("*") if f.suffix.lower() in SUFFIXES)
    docs = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("# ")), f.stem)
        docs.append({"title": title, "text": text, "source": str(f.relative_to(root))})

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(base_url=args.url, headers=headers, timeout=120) as client:
        for start in range(0, len(docs), args.batch):
            resp = client.post("/v1/rag/ingest", json={"documents": docs[start:start + args.batch]})
            resp.raise_for_status()
            for r in resp.json()["ingested"]:
                print(f"{r['status']:>9}  {r['chunks']:>3} chunks  {r['title']}")


if __name__ == "__main__":
    main()
