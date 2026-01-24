import streamlit as st
from PIL import Image
from ndid.embedding import embed_image
from ndid.ndid_logic import ndid_decision
from pathlib import Path
import uuid
from ndid.online_update import add_non_duplicate
from ndid.guard import phash_guard

NON_DUP_DIR = Path("data/non_duplicates")
NON_DUP_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="NDID", layout="wide")
st.title("NDID — Image Duplicate Detection")

st.sidebar.header("NDID Thresholds")

clip_th = st.sidebar.slider(
    "CLIP similarity threshold",
    min_value=0.70,
    max_value=0.99,
    value=0.85,
    step=0.01
)

phash_th = st.sidebar.slider(
    "pHash distance threshold",
    min_value=0,
    max_value=20,
    value=5,
    step=1
)

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Query Image", width=300)

    with st.spinner("Running NDID pipeline..."):
        q_emb = embed_image(img)
        decision, best_phash, results = ndid_decision(
            q_emb, img, clip_th, phash_th
        )

    st.subheader("Final Decision")

    if "pHash" in decision:
        st.success(f"🟢 {decision}")
    elif "CLIP" in decision:
        st.warning(f"🟡 {decision}")
    else:
        is_dup, dup_path = phash_guard(img, NON_DUP_DIR, threshold=0)

        if is_dup:
            st.success(f"🟢 DUPLICATE")
        else:
            st.error("🔴 NOT DUPLICATE")

            fname = f"nd_{uuid.uuid4().hex[:8]}.jpg"
            save_path = NON_DUP_DIR / fname
            img.save(save_path, quality=95)

            meta_row = {
                "image_path": str(save_path),
                "group_id": fname.split(".")[0],
                "transform": "original",
                "is_original": 1,
                "blurred": 0,
                "brightened": 0,
                "contrasted": 0,
                "cropped": 0,
                "original": 1
            }

            add_non_duplicate(
                embedding=q_emb,
                image_path=str(save_path),
                meta_row=meta_row
            )

            st.success("Non-duplicate image added to NDID database.")

    st.write(f"Best pHash distance: {best_phash}")

    st.subheader("Top Matches (Similarity + pHash)")

    if len(results) == 0:
        st.warning("No valid matches found (missing files or filtered out).")
    else:
        n_cols = min(len(results), 5)
        cols = st.columns(n_cols)

        for col, r in zip(cols, results):
            with col:
                st.image(r["image_path"], width=200)
                st.caption(
                    f"sim = {r['similarity']:.2f}\n"
                    f"pHash = {r['phash_distance']}"
                )
