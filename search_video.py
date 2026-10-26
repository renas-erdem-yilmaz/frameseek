from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

index_path = Path("indexes/test/index.npz")


def main():
    if not index_path.exists():
        raise RuntimeError("Index missing. Run index_video.py first.")

    with np.load(index_path, allow_pickle=False) as index:
        embeddings = index["embeddings"]
        timestamps = index["timestamps"]
        frame_paths = index["frame_paths"]
        model_name = index["model_name"].item()

    print("Loading model on CPU...")
    model = SentenceTransformer(model_name, device="cpu")

    print("\nDescribe a scene in English. Type 'quit' to exit.")

    while True:
        query = input("\nSearch: ").strip()

        if query.lower() == "quit":
            break

        if not query:
            continue

        query_embedding = model.encode(
            query,
            normalize_embeddings=True,
        )

        # Compare the description with every saved frame.
        scores = embeddings @ query_embedding
        best_matches = np.argsort(scores)[::-1][:5]

        print("\nClosest matching frames:")

        for rank, position in enumerate(best_matches, start=1):
            seconds = int(timestamps[position])
            minutes, remaining_seconds = divmod(seconds, 60)

            print(
                f"{rank}. {minutes:02d}:{remaining_seconds:02d}"
                f" | Similarity: {scores[position]:.3f}"
                f" | {frame_paths[position]}"
            )


if __name__ == "__main__":
    main()