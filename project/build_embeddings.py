# build_embeddings.py
import pandas as pd
from txtai import Embeddings


def build(
    csv_path: str = "final_data.csv",
    out_path: str = "embeddings.tar.gz",
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
        metadata = {
            "title": str(row["title"]),
            "domain": str(row["domain"]),
            "source": str(row["source"]),
        }
        records.append((i, text, metadata))

    embeddings = Embeddings(
        path= "sentence-transformers/all-MiniLM-L6-v2",
        # content=True, When RAG is used
        function="sentence",
        # graph= {'domain':'domain'} ChatGPT says deadend and useless
    )
    embeddings.index(records)
    embeddings.save(out_path)

    print(f"Saved embeddings to {out_path} with {len(records)} documents.")

if __name__ == "__main__":
    build()
