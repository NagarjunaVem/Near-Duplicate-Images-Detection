import torch
from transformers import CLIPModel, CLIPProcessor
from config import DEVICE

_model = None
_processor = None

def load_clip():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(DEVICE)
        _model.eval()
        _processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    return _model, _processor