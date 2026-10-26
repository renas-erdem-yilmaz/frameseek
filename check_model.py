from pathlib import Path

from PIL import Image
from sentence_transformers import SentenceTransformer

# Explicitly use the CPU.
print("Loading model...")
model = SentenceTransformer("clip-ViT-B-32", device="cpu")

frame_paths = sorted(Path("frames/test").glob("*.jpg"))

if not frame_paths:
    raise RuntimeError("No frames found. Run extract_frames.py first.")

# Open one frame and convert it into an embedding.
with Image.open(frame_paths[0]) as image:
    image_embedding = model.encode(image.convert("RGB"))

# Convert a description into an embedding.
text_embedding = model.encode("a person standing outside")

print("Device:", model.device)
print("Frame:", frame_paths[0].name)
print("Image embedding shape:", image_embedding.shape)
print("Text embedding shape:", text_embedding.shape)
print("Model check complete!")