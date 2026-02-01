# build_embeddings.py
import pandas as pd
import txtai


def build(
    csv_path: str = "final_data.csv",
    out_path: str = "embeddings.tar.gz",
    encoding: str = "latin1",
):
    df = pd.read_csv(csv_path, encoding=encoding)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    df = df.dropna(subset=["title", "text", "domain", "source"])

    # Create documents to embed
    # Option A: embed text only
    documents = df["text"].tolist()

    # Option B: embed title + text (often better)
    # documents = (df["title"].astype(str) + " — " + df["text"].astype(str)).tolist()

    embeddings = txtai.Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
    embeddings.index(documents)
    embeddings.save(out_path)

    print(f"Saved embeddings to {out_path} with {len(documents)} documents.")


if __name__ == "__main__":
    build()
