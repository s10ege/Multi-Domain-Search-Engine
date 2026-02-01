# cli.py
import sys
from search_engine import load_resources, search

def main():
    resources = load_resources(
        csv_path="final_data.csv",
        embeddings_path="embeddings.tar.gz",
        encoding="latin1",
    )

    print("Semantic Search CLI (txtai)")
    print("Type a query, or 'exit' to quit.\n")

    while True:
        query = input("Query> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        try:
            top_k_str = input("Top K (default 5)> ").strip()
            top_k = int(top_k_str) if top_k_str else 5
        except ValueError:
            top_k = 5

        results = search(resources, query, top_k=top_k)

        print("\nResults:")
        for i, r in enumerate(results, start=1):
            print(f"\n[{i}] {r['title']}")
            print(f"    Domain: {r['domain']} | Source: {r['source']} | Score: {r.get('score', 0):.4f}")
            text = r["text"]
            preview = (text[:350] + "...") if len(text) > 350 else text
            print(f"    {preview}")

        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
