# client.py
# Command Line Interface (CLI) entry point for the search application.
# Handles user interaction, menu navigation, and calls the appropriate search engine functions.

from __future__ import annotations

from .search_engine import (
    load_resources,
    close_resources,
    search,
    rag_answer,
    more_like_this,
)

# Helper function to interactively select a domain from the available options.
# Only ONE domain can be selected at a time (selection is mandatory).
def choose_domain(domains: list[str]) -> str:
    unique_domains = sorted(set(d.strip().lower() for d in domains))

    print("\nSelect a domain:")
    for i, d in enumerate(unique_domains, start=1):
        print(f"  {i}. {d}")

    while True:
        choice = input("Domain number: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(unique_domains):
                return unique_domains[idx - 1]
        print("Invalid choice. You must select one domain.")


def print_menu():
    print(
        """
--- MENU ---
1. Search
2. Deep Research (RAG / Answer Mode)
3. Find similar to a result
4. Change domain filter
5. Show current domain
6. Change Top-K
7. Quit
"""
    )


# Main application loop.
# Initializes resources, displays the menu, and routes user choices to search logic.
def main():
    resources = load_resources()

    # Force user to select a domain at startup (only ONE domain allowed)
    print("Semantic Search CLI (txtai)\n")
    print("Please select your initial domain:")
    domain_filter = choose_domain(resources.domains)
    print(f"\nDomain set to: {domain_filter}\n")
    
    top_k = 5
    last_results = []

    try:
        while True:
            print_menu()
            choice = input("Choose option: ").strip()
            # 1️⃣ SEARCH: Basic semantic search execution
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

            # 2️⃣ DEEP RESEARCH (RAG / ANSWER MODE): Generates answers from search results using LLM
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
                    print("\nContext related papers:")
                    for i, r in enumerate(last_results, start=1):
                        print(
                            f"  [{i}] {r['title']} (Domain: {r['domain']}, Source: {r['source']})"
                        )

                print("\n" + "-" * 60)

            # 3️⃣ FIND SIMILAR: Finds documents similar to a previously selected result
            elif choice == "3":
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

            # 4️⃣ CHANGE DOMAIN
            elif choice == "4":
                domain_filter = choose_domain(resources.domains)
                print(f"Domain set to: {domain_filter}")

            # 5️⃣ SHOW CURRENT DOMAIN
            elif choice == "5":
                print(f"Current domain filter: {domain_filter}")

            # 6️⃣ CHANGE TOP-K
            elif choice == "6":
                value = input("Enter new Top-K (e.g. 5, 10): ").strip()
                if value.isdigit() and int(value) > 0:
                    top_k = int(value)
                    print(f"Top-K set to {top_k}")
                else:
                    print("Invalid Top-K value.")

            # 7️⃣ QUIT
            elif choice == "7":
                print("Goodbye.")
                break
            else:
                print("Invalid menu option.")
    finally:
        # Ensure resources are properly closed on exit
        close_resources(resources)


if __name__ == "__main__":
    main()
