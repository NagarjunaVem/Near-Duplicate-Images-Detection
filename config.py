from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

EMB_PATH = DATA_DIR / "clip_embeddings.npy"
META_PATH = DATA_DIR / "clip_embeddings_meta.csv"
FAISS_PATH = DATA_DIR / "ndid_faiss.index"

NON_DUP_DIR = Path("data/non_duplicates")
NON_DUP_DIR.mkdir(parents=True, exist_ok=True)


TOP_K = 5
CLIP_THRESHOLD = 0.85
PHASH_THRESHOLD = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
