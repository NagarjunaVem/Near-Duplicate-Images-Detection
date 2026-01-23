import streamlit as st
from PIL import Image
from ndid.embedding import embed_image
from ndid.ndid_logic import ndid_decision
from ndid.faiss_index import load_faiss
from config import TOP_K

st.set_page_config(page_title="NDID", layout="wide")
st.title("NDID — Image Duplicate Detection")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Query Image", width=300)

    with st.spinner("Running NDID..."):
        q_emb = embed_image(img)
        decision, sims, idxs, phash_dist = ndid_decision(q_emb, img)

    st.subheader("Decision")
    if "DUPLICATE" in decision:
        st.error(decision)
    else:
        st.success(decision)

    st.write(f"Best pHash distance: {phash_dist}")
    st.write(f"Top CLIP similarity: {float(sims[1]):.3f}")

    st.subheader("Top Matches")
    _, _, meta = load_faiss()

    cols = st.columns(TOP_K - 1)
    for i in range(1, TOP_K):
        with cols[i - 1]:
            st.image(
                Image.open(meta.iloc[idxs[i]]["image_path"]),
                use_column_width=True
            )
            st.caption(f"sim={sims[i]:.2f}")