import faiss
import numpy as np
import pandas as pd
from config import EMB_PATH, META_PATH, FAISS_PATH

_index = None
_embeddings = None
_meta = None

def load_faiss():
    global _index, _embeddings, _meta

    if _index is None:
        _embeddings = np.load(EMB_PATH).astype("float32")
        _meta = pd.read_csv(META_PATH)
        _index = faiss.read_index(str(FAISS_PATH))

    return _index, _embeddings, _meta
