import sys
from PIL import Image

if __name__ == '__main__':
    img_path = sys.argv[1]
    img = Image.open(img_path)
    img = img.resize((300, 300), resample=Image.Resampling.BILINEAR)
    img.save(img_path)