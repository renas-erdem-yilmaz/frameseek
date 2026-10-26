from pathlib import Path

import easyocr
import torch


def main():
    # Limit CPU parallelism and keep the GPU unused.
    torch.set_num_threads(2)
    reader = easyocr.Reader(["en"], gpu=False)

    frames = sorted(Path("frames/test").glob("*.jpg"))

    if not frames:
        raise RuntimeError("No frames found.")

    # Test just five frames initially.
    for frame in frames:
        print(f"\n--- {frame.name} ---")

        results = reader.readtext(str(frame), workers=0, canvas_size=960)

        if not results:
            print("No text detected.")

        for _, text, confidence in results:
            print(f"{text} | OCR score: {confidence:.2f}")


if __name__ == "__main__":
    main()