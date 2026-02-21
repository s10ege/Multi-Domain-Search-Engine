# build_embeddings.py
# Script to build and save txtai embeddings from a CSV dataset.

from pathlib import Path

import pandas as pd
from txtai import Embeddings


def build(
    csv_path: str = "final_data.csv",
    out_path: str = "embeddings",
    encoding: str = "latin1",           # Latin-1 is used to encode special characters in the CSV file
):
    df = pd.read_csv(csv_path, encoding=encoding)
    # Remove unnamed columns if they exist
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Drop rows with missing values in the specified columns and reset the index
    df = df.dropna(subset=["title", "text", "domain", "source"]).reset_index(drop=True)

    # Prepare records for indexing
    records = []
    for i, row in df.iterrows():
        text = str(row["text"])
        record = {
            "text": text,
            "title": str(row["title"]),
            # Domain is treated different because, each domain will be searched individually.
            "domain": str(row["domain"]).strip().lower(),   
            "source": str(row["source"]),
        }
        records.append((i, record))

    # Feature Detection for Graph Search:
    # The application supports advanced semantic graph features, which require optional dependencies
    # ('networkx' and 'grandcypher'). This block acts as a safety toggle:
    # 1. We try to import them to checks if they are installed.
    # 2. If successful, we configure 'graph_config' to enable graph construction in txtai.
    # 3. If they are missing (ImportError), we catch the error and skip graph configuration.
    #    This ensures the script continues with standard vector search instead of crashing.
    graph_config = None
    try:
        import networkx  # noqa: F401
        import grandcypher  # noqa: F401

        graph_config = {"copyattributes": ["domain", "source"]}
    except Exception:
        graph_config = None

    embeddings_config = {
        # It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
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
