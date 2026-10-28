import hashlib
import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="FrameSeek", layout="wide")

MODEL_NAME = "clip-ViT-B-32"
INTERVAL = 2
BATCH_SIZE = 8


@st.cache_resource
def load_model():
    # Heavy imports happen only when this function is called.
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource
def load_ocr():
    import easyocr
    import torch

    torch.set_num_threads(2)
    return easyocr.Reader(["en"], gpu=False)


def label(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def process_video(video, frames_dir, index_dir, include_text):
    import time

    import cv2
    import numpy as np
    from PIL import Image

    started = time.perf_counter()
    ready_path = index_dir / "ready.json"
    ready_path.unlink(missing_ok=True)

    reader = load_ocr() if include_text else None
    capture = cv2.VideoCapture(str(video))
    records = []

    status = st.empty()
    progress = st.progress(0.0)

    try:
        if not capture.isOpened():
            raise RuntimeError("Could not open this video.")

        count = 0

        while True:
            seconds = count * INTERVAL
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            success, frame = capture.read()

            if not success:
                break

            frame_path = frames_dir / f"frame_{seconds:06d}s.jpg"

            # Save smaller thumbnails instead of full-resolution frames.
            height, width = frame.shape[:2]
            scale = min(1.0, 720 / max(height, width))

            thumbnail = cv2.resize(
                frame,
                (
                    max(1, round(width * scale)),
                    max(1, round(height * scale)),
                ),
            )

            if not cv2.imwrite(str(frame_path), thumbnail):
                raise RuntimeError(f"Could not save {frame_path}")

            detections = []

            if reader is not None:
                for box, text, score in reader.readtext(
                    frame,
                    workers=0,
                    canvas_size=960,
                ):
                    detections.append({
                        "text": text,
                        "score": float(score),
                        "box": [
                            [float(x), float(y)] for x, y in box
                        ],
                    })

            records.append({
                "timestamp": seconds,
                "frame": str(frame_path),
                "detections": detections,
            })

            count += 1
            status.info(
                f"Extracting frames"
                f"{' and reading text' if include_text else ''}: "
                f"{count} frames, through {label(seconds)}"
            )

    finally:
        capture.release()

    if not records:
        raise RuntimeError("No readable frames found.")

    extraction_seconds = time.perf_counter() - started
    embedding_started = time.perf_counter()

    status.info("Loading the visual model...")
    model = load_model()
    batches = []

    for start in range(0, len(records), BATCH_SIZE):
        batch_records = records[start:start + BATCH_SIZE]
        images = []

        for record in batch_records:
            with Image.open(record["frame"]) as image:
                images.append(image.convert("RGB"))

        try:
            batch = model.encode(
                images,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            batches.append(batch)
        finally:
            for image in images:
                image.close()

        completed = min(start + BATCH_SIZE, len(records))
        progress.progress(completed / len(records))
        status.info(
            f"Creating visual embeddings: {completed}/{len(records)}"
        )

    np.save(
        index_dir / "embeddings.npy",
        np.concatenate(batches, axis=0),
    )

    (index_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )

    metadata = {
        "model": MODEL_NAME,
        "interval": INTERVAL,
        "has_text": include_text,
        "frames": len(records),
        "extraction_seconds": round(extraction_seconds, 1),
        "embedding_seconds": round(
            time.perf_counter() - embedding_started, 1
        ),
    }

    # Mark the index as ready only after all outputs are saved.
    ready_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    status.success("Video ready to search!")


st.title("FrameSeek")
st.caption("Search visible scenes or words appearing on screen.")

upload = st.file_uploader(
    "Upload an MP4",
    type=["mp4"],
    key="visual_upload",
)

if upload is None:
    st.info("Upload a short video to begin.")
    st.stop()

video_bytes = upload.getvalue()
video_id = hashlib.sha256(video_bytes).hexdigest()

video_dir = Path("videos") / video_id

# Separate optimized indexes from the previous version.
frames_dir = Path("frames") / video_id / "fast_v1"
index_dir = Path("indexes") / video_id / "fast_v1"

for folder in (video_dir, frames_dir, index_dir):
    folder.mkdir(parents=True, exist_ok=True)

video_path = video_dir / "video.mp4"

if not video_path.exists():
    video_path.write_bytes(video_bytes)

include_text = st.checkbox(
    "Read on-screen text (slower)",
    value=False,
)

st.caption(
    "CPU mode · one frame every two seconds · "
    "OCR is optional and uses reduced detection resolution."
)

if st.button("Process video", type="primary"):
    try:
        process_video(
            video_path,
            frames_dir,
            index_dir,
            include_text,
        )
        st.session_state[f"visual_play_{video_id}"] = 0
    except Exception as error:
        st.error(f"Processing failed: {error}")
        st.stop()

player = st.empty()
ready_path = index_dir / "ready.json"

if not ready_path.exists():
    player.video(str(video_path))
    st.info("Click Process video to prepare search.")
    st.stop()

metadata = json.loads(ready_path.read_text(encoding="utf-8"))
records = json.loads(
    (index_dir / "records.json").read_text(encoding="utf-8")
)

st.caption(
    f"{metadata['frames']} frames indexed · "
    f"Extraction/OCR: {metadata['extraction_seconds']}s · "
    f"Embeddings/model loading: {metadata['embedding_seconds']}s"
)

modes = ["Visual description"]

if metadata["has_text"]:
    modes.append("On-screen text")

mode = st.radio("Search mode", modes, horizontal=True)

query = st.text_input(
    "Search",
    placeholder="Describe a scene in English",
    key=f"visual_query_{video_id}",
).strip()

results = []

if query:
    if mode == "Visual description":
        import numpy as np

        embeddings = np.load(
            index_dir / "embeddings.npy",
            allow_pickle=False,
        )

        with st.spinner("Searching scenes..."):
            model = load_model()
            query_embedding = model.encode(
                query,
                normalize_embeddings=True,
            )
            scores = embeddings @ query_embedding

        for position in np.argsort(scores)[::-1][:5]:
            results.append((
                records[int(position)],
                f"Similarity: {scores[position]:.3f}",
            ))

        st.caption(
            "Similarity scores are not confidence percentages. "
            "Matches may be irrelevant if the scene is absent."
        )

    else:
        for record in records:
            matches = [
                item["text"]
                for item in record["detections"]
                if query.casefold() in item["text"].casefold()
            ]

            if matches:
                results.append((record, " / ".join(matches)))

        st.caption(
            "A word appearing on screen does not establish "
            "that a particular event occurred."
        )

    if not results:
        st.info("No matching text detected.")

    for i, (record, detail) in enumerate(results):
        with st.container(border=True):
            thumbnail, description = st.columns([1, 2])

            with thumbnail:
                st.image(record["frame"], width=240)

            with description:
                time_label = label(record["timestamp"])
                st.write(f"**{time_label}**")
                st.write(detail)

                if st.button(
                    f"Jump to {time_label}",
                    key=f"visual_jump_{video_id}_{i}",
                ):
                    st.session_state[f"visual_play_{video_id}"] = (
                        record["timestamp"]
                    )

player.video(
    str(video_path),
    start_time=st.session_state.get(f"visual_play_{video_id}", 0),
)