# Near-Duplicate Images Detection (NDID)

A Near-Duplicate Image Detection (NDID) system built using CLIP embeddings, FAISS similarity search, and perceptual hashing (pHash), wrapped in an interactive Streamlit application.

The system determines whether an uploaded image is:

- an exact duplicate
- a near-duplicate (crop, blur, brightness/contrast changes)
- or unique

---

## Features

- CLIP-based semantic embeddings (openai/clip-vit-base-patch32)
- FAISS for fast Top-K nearest neighbor search
- Perceptual Hashing (pHash) for pixel-level duplicate detection
- Hybrid decision logic (CLIP + pHash)
- Recall@K, Precision, F1-score evaluation
- Streamlit web UI with live threshold tuning
- Modular, production-style Python code

---

## Directory Structure

```
Near-Duplicate-Images-Detection/
├── .gitignore
├── README.md
├── app.py
├── config.py
├── ndid/
│   ├── clip_model.py
│   ├── embedding.py
│   ├── faiss_index.py
│   ├── ndid_logic.py
|   ├── phash.py
│   └── online_update.py
├── data/
│   ├── non_duplicates/
│   ├── clip_embeddings.npy
│   ├── clip_embeddings_meta.csv
│   └── ndid_faiss.index
├── notebooks/
│   ├── classification.ipynb
│   ├── data_input.ipynb
│   ├── embedding.ipynb
│   ├── embeddings_visualisation.ipynb
│   ├── evaluation.ipynb
│   ├── evaluation_faiss.ipynb
│   ├── faiss.ipynb
│   ├── hashing.ipynb
│   ├── preprocessing.ipynb
│   ├── preprocessing_visualisation.ipynb
│   └── similarity.ipynb
└── requirements.txt
```

---

## NDID Pipeline

Image Upload
→ CLIP Embedding
→ FAISS Top-K Search
→ pHash Comparison
→ Hybrid Decision
→ Duplicate / Not-Duplicate

Decision rules:

- pHash ≤ threshold → Exact / near-exact duplicate
- CLIP similarity ≥ threshold → Semantic near-duplicate
- Otherwise → Not duplicate

---

## Configuration

Edit config.py:

TOP_K = 10  
CLIP_THRESHOLD = 0.85  
PHASH_THRESHOLD = 5

Thresholds can also be adjusted live in the Streamlit UI.

---

## Running the Streamlit App

1. Install dependencies  
   pip install -r requirements.txt

2. Required data files  
   Create a data/ directory containing:

- clip_embeddings.npy
- clip_embeddings_meta.csv
- ndid_faiss.index

3. Launch the app  
   streamlit run app.py

Then open http://localhost:8501

---

## Online Ingestion (Important)

When an uploaded image is classified as **NOT DUPLICATE**:

- The image is saved to `data/non_duplicates/`
- Its CLIP embedding is appended to `clip_embeddings.npy`
- A new row is appended to `clip_embeddings_meta.csv`
- The FAISS index is updated **in memory and on disk**

This ensures:

- Re-uploading the same image **immediately** results in a duplicate
- No application restart is required
- Cache coherence is maintained

---

## Evaluation & Notebooks

- `preprocessing.ipynb` — Image filtering & cleaning
- `embedding.ipynb` — CLIP embedding extraction
- `faiss.ipynb` — FAISS index creation
- `similarity.ipynb` — Similarity exploration
- `evaluation_faiss.ipynb` — Recall@K computation
- `classification.ipynb` — Duplicate classification
- `hashing.ipynb` — pHash analysis

---

## Metrics

- **Recall@K** — Retrieval quality
- **Precision** — False positive control
- **F1-score** — Balanced NDID performance

Ground truth is derived from `group_id`.

---

## Why CLIP + FAISS + pHash?

- **CLIP** — Robust semantic similarity across transformations
- **FAISS** — Fast, scalable nearest-neighbor retrieval
- **pHash** — Reliable exact and near-exact duplicate detection

Together, they form a **production-grade NDID system**.

---

## Notes

- NDID decision logic is **pure** (no side effects)
- Disk writes and DB mutation occur only in the app/controller layer
- FAISS index, embeddings, and metadata are cached in memory
- Missing image paths are handled safely (no crashes)

---

## Future Work

- Batch upload & ingestion
- REST API (FastAPI)
- Automatic threshold tuning
- Hard-negative mining
- Model fine-tuning
- GPU FAISS support
- Multi-user concurrency locks

---

## Author

Research-grade NDID system for learning, experimentation, and deployment.
