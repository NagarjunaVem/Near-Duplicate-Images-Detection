import imagehash
from PIL import Image

def compute_phash(img: Image.Image):
    return imagehash.phash(img)

def phash_distance(img1, img2):
    return compute_phash(img1) - compute_phash(img2)