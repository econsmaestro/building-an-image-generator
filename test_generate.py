import torch
import base64
from model_cgan import make_image_v2, CIFAR10_CLASSES
from PIL import Image
import io

prompts = ["airplane", "cat", "dog", "horse", "ship", "truck"]

for prompt in prompts:
    b64 = make_image_v2(prompt)
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    fname = f"test_{prompt}.png"
    img.save(fname)
    print(f"Saved {fname}")

print("\nDone! Open the PNG files to see the generated images.")
