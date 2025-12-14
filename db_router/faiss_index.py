import faiss
import numpy as np

# Load the index
index = faiss.read_index("router_index.faiss")
print(f"Index dimension: {index.d}")
print(f"Number of vectors in index: {index.ntotal}")

index.remove_ids(np.array([0, 1, 2], dtype=np.int64))
index = faiss.read_index("router_index.faiss")
print(f"Index dimension: {index.d}")
print(f"Number of vectors in index: {index.ntotal}")

# Depending on the index type, you might have other properties like
# index.nlist for IVF indexes, or index.M for HNSW indexes.
