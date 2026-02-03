# build_embeddings.py
from pathlib import Path

import pandas as pd
from txtai import Embeddings


def build(
    csv_path: str = "final_data.csv",
    out_path: str = "embeddings",
    encoding: str = "latin1",
):
    df = pd.read_csv(csv_path, encoding=encoding)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    df = df.dropna(subset=["title", "text", "domain", "source"]).reset_index(drop=True)

    records = []
    for i, row in df.iterrows():
        text = str(row["text"])
        record = {
            "text": text,
            "title": str(row["title"]),
            "domain": str(row["domain"]).strip().lower(),
            "source": str(row["source"]),
        }
        records.append((i, record))

    graph_config = None
    try:
        import networkx  # noqa: F401
        import grandcypher  # noqa: F401

        graph_config = {"copyattributes": ["domain", "source"]}
    except Exception:
        graph_config = None

    embeddings_config = {
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "content": True,
        "scoring": {"method": "bm25", "terms": True},
        "function": "sentence",
    }
    if graph_config:
        embeddings_config["graph"] = graph_config

    embeddings = Embeddings(embeddings_config)
    embeddings.index(records)
    embeddings.save(out_path)

    embeddings.close()

    print(f"Saved embeddings to {out_path} with {len(records)} documents.")

if __name__ == "__main__":
    build()
