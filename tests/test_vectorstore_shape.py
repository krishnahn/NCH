import numpy as np
import sys
import os

# Ensure project package imports work when running test directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.vectorstore import ChromaVectorStore
except Exception as e:
    print("ERROR importing ChromaVectorStore:", e)
    raise

print("Creating ChromaVectorStore instance without running __init__ (avoid model load)")
store = object.__new__(ChromaVectorStore)
store.index = None
store.metadata = []

# Provide minimal stubs for Chroma collection/client to avoid running __init__
class _DummyCollection:
    def __init__(self, store_ref):
        self._store = store_ref
    def add(self, **kwargs):
        metadatas = kwargs.get("metadatas")
        if metadatas:
            if isinstance(metadatas, dict):
                self._store.metadata.append(metadatas)
            else:
                self._store.metadata.extend(metadatas)

class _DummyClient:
    def persist(self):
        return None

store.collection = _DummyCollection(store)
store.client = _DummyClient()

def run_test(arr, desc):
    print(f"\n--- {desc} ---")
    try:
        store.add_embeddings(arr)
        print("Success. metadata length:", len(store.metadata))
    except Exception as e:
        print("Exception:", type(e).__name__, e)

# Empty array
run_test(np.array([]), "Empty array")

# 1-D vector
run_test(np.array([1.0, 2.0, 3.0]), "1-D vector")

# 2-D vectors
run_test(np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]), "2-D vectors")

print('\nTest script finished.')
