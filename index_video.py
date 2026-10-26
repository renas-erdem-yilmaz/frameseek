from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

frames_folder = Path("frames/test")
index_folder = Path("indexes/test")
model_name = "clip-ViT-B-32"


def main():
    frame_paths = sorted(frames_folder.glob("*.jpg"))

    if not frame_paths:
        raise RuntimeError("No frames found. Run extract_frames.py first.")

    print("Loading model on CPU...")
    model = SentenceTransformer(model_name, device="cpu")

    embeddings = []
    timestamps = []

    for i, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as image:
            embedding = model.encode(
                image.convert("RGB"),
                normalize_embeddings=True,
            )

        embeddings.append(embedding)

        # Example: frame_000012s.jpg → 12 seconds
        timestamp = int(frame_path.stem.removeprefix("frame_").removesuffix("s"))
        timestamps.append(timestamp)

        print(f"Processed {i + 1}/{len(frame_paths)} frames")

    index_folder.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        index_folder / "index.npz",
        embeddings=np.stack(embeddings),
        timestamps=np.array(timestamps),
        frame_paths=np.array([str(path) for path in frame_paths]),
        model_name=np.array(model_name),
    )

    print(f"Done! Index saved to {index_folder / 'index.npz'}")


if __name__ == "__main__":
    main()