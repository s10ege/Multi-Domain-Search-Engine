from __future__ import annotations

from search_engine import (
    load_resources,
    close_resources,
    search,
    rag_answer,
    summarize_results,
    more_like_this,
    rerank_search,
    cluster_results,
)


def choose_domain(domains: list[str]) -> str | None:
    unique_domains = sorted(set(d.strip().lower() for d in domains))

    print("\nSelect a domain (or press Enter for ALL):")
    for i, d in enumerate(unique_domains, start=1):
        print(f"  {i}. {d}")

    while True:
        choice = input("Domain number (or Enter): ").strip()
        if choice == "":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(unique_domains):
                return unique_domains[idx - 1]
        print("Invalid choice.")


def print_menu():
    print(
        """
--- MENU ---
1. Search
2. RAG / Answer mode
3. Summarize top-k
4. Find similar to a result
5. Search (reranked)
6. Show clusters (optional)
7. Change domain filter
8. Show current domain
9. Change Top-K
10. Clear domain filter
11. Quit
"""
    )


def main():
    resources = load_resources()

    domain_filter = None
    top_k = 5
    last_results = []

    print("Semantic Search CLI (txtai)\n")

    try:
        while True:
            print_menu()
            choice = input("Choose option: ").strip()
            # 1️⃣ SEARCH
            if choice == "1":
                query = input("\nQuery> ").strip()
                if not query:
                    continue

                results = search(
                    resources,
                    query,
                    top_k=top_k,
                    domain=domain_filter,
                )

                last_results = results
                print("\nResults:")
                if not results:
                    print("  No results found.")
                for i, r in enumerate(results, start=1):
                    print(f"\n[{i}] {r['title']}")
                    print(
                        f"    Domain: {r['domain']} | "
                        f"Source: {r['source']} | "
                        f"Score: {r['score']:.4f}"
                    )
                    preview = r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"]
                    print(f"    {preview}")

                print("\n" + "-" * 60)

            # 2️⃣ RAG / ANSWER MODE
            elif choice == "2":
                question = input("\nQuestion> ").strip()
                if not question:
                    continue

                output = rag_answer(
                    resources,
                    question,
                    top_k=min(top_k, 5),
                    domain=domain_filter,
                )

                answer = output.get("answer", "")
                last_results = output.get("results", [])

                print("\nAnswer:")
                print(answer if answer else "  No answer generated.")

                if last_results:
                    print("\nContext results:")
                    for i, r in enumerate(last_results, start=1):
                        print(
                            f"  [{i}] {r['title']} (Domain: {r['domain']}, Source: {r['source']})"
                        )

                print("\n" + "-" * 60)

            # 3️⃣ SUMMARIZE TOP-K
            elif choice == "3":
                query = input("\nQuery to summarize> ").strip()
                if not query:
                    continue

                output = summarize_results(
                    resources,
                    query,
                    top_k=top_k,
                    domain=domain_filter,
                )

                summary = output.get("summary", "")
                last_results = output.get("results", [])

                print("\nSummary:")
                print(summary if summary else "  No summary generated.")

                if last_results:
                    print("\nSummarized results:")
                    for i, r in enumerate(last_results, start=1):
                        print(
                            f"  [{i}] {r['title']} (Domain: {r['domain']}, Source: {r['source']})"
                        )

                print("\n" + "-" * 60)

            # 4️⃣ FIND SIMILAR TO A RESULT
            elif choice == "4":
                if not last_results:
                    print("Run a search or RAG first to select a result.")
                    continue

                print("\nSelect a result to find similar documents:")
                for i, r in enumerate(last_results, start=1):
                    print(f"  {i}. {r['title']} (Domain: {r['domain']})")

                choice_idx = input("Result number> ").strip()
                if not choice_idx.isdigit():
                    print("Invalid selection.")
                    continue

                idx = int(choice_idx)
                if idx < 1 or idx > len(last_results):
                    print("Invalid selection.")
                    continue

                seed_index = last_results[idx - 1]["index"]
                results = more_like_this(
                    resources,
                    seed_index,
                    top_k=top_k,
                    domain=domain_filter,
                )
                last_results = results

                print("\nSimilar results:")
                if not results:
                    print("  No results found.")
                for i, r in enumerate(results, start=1):
                    print(f"\n[{i}] {r['title']}")
                    print(
                        f"    Domain: {r['domain']} | "
                        f"Source: {r['source']} | "
                        f"Score: {r['score']:.4f}"
                    )
                    preview = r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"]
                    print(f"    {preview}")

                print("\n" + "-" * 60)

            # 5️⃣ SEARCH (RERANKED)
            elif choice == "5":
                query = input("\nQuery> ").strip()
                if not query:
                    continue

                results = rerank_search(
                    resources,
                    query,
                    top_k=top_k,
                    domain=domain_filter,
                )
                last_results = results

                print("\nReranked Results:")
                if not results:
                    print("  No results found.")
                for i, r in enumerate(results, start=1):
                    print(f"\n[{i}] {r['title']}")
                    print(
                        f"    Domain: {r['domain']} | "
                        f"Source: {r['source']} | "
                        f"Score: {r['score']:.4f} | "
                        f"Rerank: {r.get('rerank_score', 0.0):.4f}"
                    )
                    preview = r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"]
                    print(f"    {preview}")

                print("\n" + "-" * 60)

            # 6️⃣ SHOW CLUSTERS (OPTIONAL)
            elif choice == "6":
                query = input("\nQuery to cluster> ").strip()
                if not query:
                    continue

                try:
                    clusters = cluster_results(
                        resources,
                        query,
                        top_k=max(top_k, 10),
                        domain=domain_filter,
                    )
                except RuntimeError as exc:
                    print(str(exc))
                    continue

                if not clusters:
                    print("No results to cluster.")
                    continue

                print("\nClusters:")
                for cluster in clusters:
                    print(f"\nCluster {cluster['cluster']}")
                    for item in cluster["items"]:
                        print(f"  - {item['title']} (Domain: {item['domain']})")

                print("\n" + "-" * 60)

            # 7️⃣ CHANGE DOMAIN
            elif choice == "7":
                domain_filter = choose_domain(resources.domains)
                if domain_filter:
                    print(f"Domain set to: {domain_filter}")
                else:
                    print("Searching across ALL domains")

            # 8️⃣ SHOW CURRENT DOMAIN
            elif choice == "8":
                print(
                    f"Current domain filter: {domain_filter if domain_filter else 'ALL'}"
                )

            # 9️⃣ CHANGE TOP-K
            elif choice == "9":
                value = input("Enter new Top-K (e.g. 5, 10): ").strip()
                if value.isdigit() and int(value) > 0:
                    top_k = int(value)
                    print(f"Top-K set to {top_k}")
                else:
                    print("Invalid Top-K value.")

            # 10️⃣ CLEAR DOMAIN
            elif choice == "10":
                domain_filter = None
                print("Domain filter cleared (ALL domains).")

            # 11️⃣ QUIT
            elif choice == "11":
                print("Goodbye.")
                break
            else:
                print("Invalid menu option.")
    finally:
        close_resources(resources)


if __name__ == "__main__":
    main()
