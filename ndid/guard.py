from pathlib import Path
from PIL import Image
import imagehash

def phash_guard(query_img, folder, threshold=0):
    qh = imagehash.phash(query_img)

    for p in Path(folder).glob("*.jpg"):
        try:
            img = Image.open(p)
            if qh - imagehash.phash(img) <= threshold:
                return True, str(p)
        except:
            continue

    return False, None
