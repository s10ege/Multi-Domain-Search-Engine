from txtai import Embeddings
from search_engine import load_resources, search

resources = load_resources()
results = search(resources, "machine learning" , top_k=10, domain="research")
print(results)
