from ndid.faiss_index import load_faiss
from ndid.phash import phash_distance
from config import TOP_K, CLIP_THRESHOLD, PHASH_THRESHOLD
from PIL import Image
from pathlib import Path

def ndid_decision(query_emb, query_img, clip_th, phash_th):
    index, _, meta = load_faiss()

    sims, idxs = index.search(query_emb, TOP_K)
    sims = sims[0]
    idxs = idxs[0]

    best_phash = 64
    decision = "NOT DUPLICATE"
    results = []

    for rank, (sim, idx) in enumerate(zip(sims[0:], idxs[0:]), start=0):
        img_path = Path(meta.iloc[idx]["image_path"])
        if not img_path.exists():
            continue

        cand_img = Image.open(img_path)
        dist = phash_distance(query_img, cand_img)
        best_phash = min(best_phash, dist)

        match_type = "NONE"
        if dist <= phash_th:
            match_type = "PHASH"
            decision = "DUPLICATE (pHash)"
        elif sim >= clip_th and decision == "NOT DUPLICATE":
            match_type = "CLIP"
            decision = "DUPLICATE (CLIP)"

        results.append({
            "rank": rank,
            "image_path": str(img_path),
            "similarity": float(sim),
            "phash_distance": int(dist),
            "match_type": match_type
        })

    return decision, best_phash, results
