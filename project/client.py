from __future__ import annotations

from search_engine import load_resources, search


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
2. Change domain filter
3. Show current domain
4. Change Top-K
5. Clear domain filter
6. Quit
"""
    )


def main():
    resources = load_resources()

    domain_filter = None
    top_k = 5

    print("Semantic Search CLI (txtai)\n")

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

        # 2️⃣ CHANGE DOMAIN
        elif choice == "2":
            domain_filter = choose_domain(resources.domains)
            if domain_filter:
                print(f"Domain set to: {domain_filter}")
            else:
                print("Searching across ALL domains")

        # 3️⃣ SHOW CURRENT DOMAIN
        elif choice == "3":
            print(
                f"Current domain filter: {domain_filter if domain_filter else 'ALL'}"
            )

        # 4️⃣ CHANGE TOP-K
        elif choice == "4":
            value = input("Enter new Top-K (e.g. 5, 10): ").strip()
            if value.isdigit() and int(value) > 0:
                top_k = int(value)
                print(f"Top-K set to {top_k}")
            else:
                print("Invalid Top-K value.")

        # 5️⃣ CLEAR DOMAIN
        elif choice == "5":
            domain_filter = None
            print("Domain filter cleared (ALL domains).")

        # 6️⃣ QUIT
        elif choice == "6":
            print("Goodbye.")
            break

        else:
            print("Invalid menu option.")


if __name__ == "__main__":
    main()
