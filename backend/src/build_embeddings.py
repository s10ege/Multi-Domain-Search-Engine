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

    # Prepare records for indexing.
    # The indexed "text" field is title + body so that title keywords contribute
    # to the semantic vector. Display text is read from the CSV at load time,
    # so the content store value doesn't affect what users see.
    records = []
    for i, row in df.iterrows():
        title = str(row["title"])
        body  = str(row["text"])
        record = {
            "text": f"{title} [SEP] {body}",
            "title": title,
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
        # bge-small-en-v1.5: retrieval-optimised, 384-dim, 512-token max.
        # Same dimensions as the previous model — FAISS config needs no change.
        "path": "BAAI/bge-small-en-v1.5",
        "content": True,
        "scoring": {"method": "bm25", "terms": True},
        "instructions": {
            # BGE models are designed with asymmetric retrieval: queries get a
            # task prefix, documents do not.
            "query": "Represent this sentence for searching relevant passages: ",
        },
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
