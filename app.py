import streamlit as st
from PIL import Image
from ndid.embedding import embed_image
from ndid.ndid_logic import ndid_decision

st.set_page_config(page_title="NDID", layout="wide")
st.title("NDID — Image Duplicate Detection")

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("NDID Thresholds")

clip_th = st.sidebar.slider(
    "CLIP similarity threshold",
    min_value=0.70,
    max_value=0.99,
    value=0.90,
    step=0.01
)

phash_th = st.sidebar.slider(
    "pHash distance threshold",
    min_value=0,
    max_value=20,
    value=5,
    step=1
)

# ---------------- IMAGE UPLOAD ----------------
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Query Image", width=300)

    with st.spinner("Running NDID pipeline..."):
        q_emb = embed_image(img)
        decision, best_phash, results = ndid_decision(
            q_emb, img, clip_th, phash_th
        )

    # ---------------- FINAL DECISION ----------------
    st.subheader("Final Decision")

    if "pHash" in decision:
        st.success(f"🟢 {decision}")
    elif "CLIP" in decision:
        st.warning(f"🟡 {decision}")
    else:
        st.error("🔴 NOT DUPLICATE")

    st.write(f"Best pHash distance: {best_phash}")

    # ---------------- TOP MATCHES ----------------
    st.subheader("Top Matches (Similarity + pHash)")

    if len(results) == 0:
        st.warning("No valid matches found (missing files or filtered out).")
    else:
        cols = st.columns(len(results))
        for col, r in zip(cols, results):
            with col:
                st.image(r["image_path"], width=200)
                st.caption(
                    f"sim = {r['similarity']:.2f}\n"
                    f"pHash = {r['phash_distance']}"
                )
