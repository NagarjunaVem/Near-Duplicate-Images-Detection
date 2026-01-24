import numpy as np
import pandas as pd
import faiss
from config import EMB_PATH, META_PATH, FAISS_PATH
from ndid.faiss_index import add_to_memory

def add_non_duplicate(embedding, image_path, meta_row):
    embeddings = np.load(EMB_PATH).astype("float32")
    meta = pd.read_csv(META_PATH)
    index = faiss.read_index(str(FAISS_PATH))

    embeddings = np.vstack([embeddings, embedding])
    meta = pd.concat([meta, pd.DataFrame([meta_row])], ignore_index=True)
    index.add(embedding)

    np.save(EMB_PATH, embeddings)
    meta.to_csv(META_PATH, index=False)
    faiss.write_index(index, str(FAISS_PATH))

    add_to_memory(embedding, meta_row)
