import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="FrameSeek", layout="wide")

MODEL_NAME = "clip-ViT-B-32"
INTERVAL = 2


@st.cache_resource
def load_model():
    torch.set_num_threads(2)
    return SentenceTransformer(MODEL_NAME, device="cpu")


@st.cache_resource
def load_ocr():
    import easyocr

    torch.set_num_threads(2)
    return easyocr.Reader(["en"], gpu=False)


def label(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def process_video(video, frames_dir, index_dir, include_text):
    ready = index_dir / "ready.json"

    # Incomplete processing must not be treated as a valid index.
    ready.unlink(missing_ok=True)

    model = load_model()
    reader = load_ocr() if include_text else None

    capture = cv2.VideoCapture(str(video))
    embeddings = []
    records = []
    status = st.empty()

    try:
        if not capture.isOpened():
            raise RuntimeError("Could not open this video.")

        count = 0

        while True:
            timestamp = count * INTERVAL
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, frame = capture.read()

            if not success:
                break

            frame_path = frames_dir / f"frame_{timestamp:06d}s.jpg"

            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Could not save frame at {timestamp}s.")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embedding = model.encode(
                Image.fromarray(rgb),
                normalize_embeddings=True,
            )
            embeddings.append(embedding)

            detections = []

            if reader is not None:
                for box, text, score in reader.readtext(frame, workers=0):
                    detections.append({
                        "text": text,
                        "score": float(score),
                        "box": [
                            [float(x), float(y)] for x, y in box
                        ],
                    })

            records.append({
                "timestamp": timestamp,
                "frame": str(frame_path),
                "detections": detections,
            })

            count += 1
            status.info(
                f"Processed {count} frames — reached {label(timestamp)}"
            )

    finally:
        capture.release()

    if not records:
        raise RuntimeError("No readable frames found in this video.")

    np.save(index_dir / "embeddings.npy", np.stack(embeddings))

    (index_dir / "records.json").write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write this last, after both indexes are saved.
    ready.write_text(
        json.dumps({
            "model": MODEL_NAME,
            "interval": INTERVAL,
            "has_text": include_text,
        }),
        encoding="utf-8",
    )

    status.success(f"Ready! Processed {len(records)} frames.")


st.title("FrameSeek")
st.caption("Upload a video, process it, and search its moments.")

upload = st.file_uploader("Upload an MP4", type=["mp4"])

if upload is None:
    st.info("Choose a short video to get started.")
    st.stop()

# Identify videos by their contents, not their filenames.
video_bytes = upload.getvalue()
video_id = hashlib.sha256(video_bytes).hexdigest()

video_dir = Path("videos") / video_id
frames_dir = Path("frames") / video_id
index_dir = Path("indexes") / video_id

for folder in (video_dir, frames_dir, index_dir):
    folder.mkdir(parents=True, exist_ok=True)

video_path = video_dir / "video.mp4"

if not video_path.exists():
    video_path.write_bytes(video_bytes)

include_text = st.checkbox(
    "Read on-screen text too (slower)",
    value=True,
)

st.caption(
    "Processing uses the CPU and samples one frame every two seconds. "
    "Start with a short clip."
)

if st.button("Process video", type="primary"):
    try:
        with st.spinner("Preparing models and processing video..."):
            process_video(
                video_path,
                frames_dir,
                index_dir,
                include_text,
            )
        st.session_state[f"play_{video_id}"] = 0
    except Exception as error:
        st.error(f"Processing failed: {error}")
        st.stop()

player = st.empty()
ready_path = index_dir / "ready.json"

if not ready_path.exists():
    player.video(str(video_path))
    st.info("Click Process video to enable searching.")
    st.stop()

metadata = json.loads(ready_path.read_text(encoding="utf-8"))
records = json.loads(
    (index_dir / "records.json").read_text(encoding="utf-8")
)

modes = ["Visual description"]

if metadata["has_text"]:
    modes.append("On-screen text")

mode = st.radio("Search mode", modes, horizontal=True)

query = st.text_input(
    "Search",
    placeholder="Describe a scene, or search a name in text mode",
    key=f"query_{video_id}",
).strip()

results = []

if query:
    if mode == "Visual description":
        embeddings = np.load(
            index_dir / "embeddings.npy",
            allow_pickle=False,
        )

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
            "Scores measure similarity, not confidence. "
            "Closest matches may still be irrelevant."
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
            "Text matches show where words appear, "
            "not whether an event happened."
        )

    if not results:
        st.info("No matching text detected.")

    for i, (record, detail) in enumerate(results):
        with st.container(border=True):
            thumbnail, description = st.columns([1, 2])

            with thumbnail:
                st.image(record["frame"], width=240)

            with description:
                time = label(record["timestamp"])
                st.write(f"**{time}**")
                st.write(detail)

                if st.button(
                    f"Jump to {time}",
                    key=f"jump_{video_id}_{i}",
                ):
                    st.session_state[f"play_{video_id}"] = (
                        record["timestamp"]
                    )

player.video(
    str(video_path),
    start_time=st.session_state.get(f"play_{video_id}", 0),
)