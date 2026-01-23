import torch
import numpy as np
from ndid.clip_model import load_clip
from config import DEVICE

def embed_image(img):
    model, processor = load_clip()
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        feat = model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.cpu().numpy().astype("float32")