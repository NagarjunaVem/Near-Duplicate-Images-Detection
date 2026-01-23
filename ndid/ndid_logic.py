from ndid.faiss_index import load_faiss
from ndid.phash import phash_distance
from config import TOP_K, CLIP_THRESHOLD, PHASH_THRESHOLD
from PIL import Image

def ndid_decision(query_emb, query_img):
    index, _, meta = load_faiss()

    sims, idxs = index.search(query_emb, TOP_K)
    sims = sims[0]
    idxs = idxs[0]

    best_phash = 64
    decision = "NOT DUPLICATE"

    for sim, idx in zip(sims[1:], idxs[1:]):
        cand_img = Image.open(meta.iloc[idx]["image_path"])
        dist = phash_distance(query_img, cand_img)
        best_phash = min(best_phash, dist)

        if dist <= PHASH_THRESHOLD:
            return "DUPLICATE (pHash)", sims, idxs, best_phash

        if sim >= CLIP_THRESHOLD:
            decision = "DUPLICATE (CLIP)"

    return decision, sims, idxs, best_phash
