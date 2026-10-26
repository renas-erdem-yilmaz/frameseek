import json
from pathlib import Path

import easyocr
import torch


def main():
    torch.set_num_threads(2)
    reader = easyocr.Reader(["en"], gpu=False)

    frames = sorted(Path("frames/test").glob("*.jpg"))

    if not frames:
        raise RuntimeError("No frames found. Run extract_frames.py first.")

    records = []

    for i, frame in enumerate(frames, start=1):
        timestamp = int(
            frame.stem.removeprefix("frame_").removesuffix("s")
        )

        detections = []

        for box, text, confidence in reader.readtext(str(frame), workers=0):
            detections.append({
                "text": text,
                "score": float(confidence),
                "box": [[float(x), float(y)] for x, y in box],
            })

        records.append({
            "timestamp": timestamp,
            "frame": str(frame),
            "detections": detections,
        })

        print(f"Processed {i}/{len(frames)}")

    output = Path("indexes/test/text.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved text index to {output}")


if __name__ == "__main__":
    main()