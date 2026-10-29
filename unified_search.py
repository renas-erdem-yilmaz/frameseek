import hashlib
import json
from pathlib import Path

import streamlit as st

st.title("FrameSeek — Combined Search")

CLIP_MODEL = "clip-ViT-B-32"
SPEECH_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


@st.cache_resource
def load_model(name):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    return SentenceTransformer(name, device="cpu")


def label(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def search_speech(records, query, folder):
    import numpy as np

    texts = [record["text"] for record in records]

    signature = hashlib.sha256(
        json.dumps(
            {"model": SPEECH_MODEL, "texts": texts},
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    path = folder / f"semantic_{signature}.npy"
    model = load_model(SPEECH_MODEL)

    if path.exists():
        embeddings = np.load(path, allow_pickle=False)
    else:
        embeddings = model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as output:
            np.save(output, embeddings)
        temporary.replace(path)

    query_embedding = model.encode(
        query, normalize_embeddings=True
    )
    scores = embeddings @ query_embedding

    return [
        {
            "time": records[int(i)]["start"],
            "text": records[int(i)]["text"],
            "score": float(scores[i]),
            "frame": None,
        }
        for i in np.argsort(scores)[::-1][:3]
    ]


video_root = Path("videos")
videos = sorted(video_root.glob("*/video.mp4"))

if not videos:
    st.info("Upload and process a video on either search page first.")
    st.stop()

video = st.selectbox(
    "Saved video",
    videos,
    format_func=lambda path: (
        f"{path.parent.name[:12]} "
        f"({path.stat().st_size / 1_000_000:.1f} MB)"
    ),
)

video_id = video.parent.name
index_root = Path("indexes") / video_id

# Prefer the optimized visual index, with the original as fallback.
visual_folder = None

for candidate in (index_root / "fast_v1", index_root):
    if (candidate / "ready.json").exists():
        visual_folder = candidate
        break

transcripts = sorted(index_root.glob("speech_*.json"))
transcript_path = None

if transcripts:
    transcript_path = st.selectbox(
        "Saved transcript",
        transcripts,
        format_func=lambda path: path.stem,
    )

if visual_folder is None and transcript_path is None:
    st.info("This video has no completed indexes. Process it first.")
    st.stop()

player = st.empty()
play_key = f"combined_play_{video_id}"

with st.form("combined_search"):
    query = st.text_input(
        "Describe the moment",
        placeholder="For example: someone explains the score",
    ).strip()

    use_visual = st.checkbox(
        "Search visible scenes",
        value=visual_folder is not None,
        disabled=visual_folder is None,
    )

    use_speech = st.checkbox(
        "Search spoken commentary",
        value=transcript_path is not None,
        disabled=transcript_path is None,
    )

    submitted = st.form_submit_button("Search")

results_key = (
    f"combined_results_{video_id}_"
    f"{transcript_path.stem if transcript_path else 'no_speech'}"
)

if submitted:
    results = {}

    if query:
        if use_visual and visual_folder is not None:
            try:
                with st.spinner("Searching visible scenes..."):
                    import numpy as np

                    metadata = json.loads(
                        (visual_folder / "ready.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    records = json.loads(
                        (visual_folder / "records.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    embeddings = np.load(
                        visual_folder / "embeddings.npy",
                        allow_pickle=False,
                    )

                    model = load_model(
                        metadata.get("model", CLIP_MODEL)
                    )
                    vector = model.encode(
                        query, normalize_embeddings=True
                    )
                    scores = embeddings @ vector

                    results["Visible scenes"] = [
                        {
                            "time": records[int(i)]["timestamp"],
                            "frame": records[int(i)]["frame"],
                            "text": "",
                            "score": float(scores[i]),
                        }
                        for i in np.argsort(scores)[::-1][:3]
                    ]

            except Exception as error:
                st.error(f"Visual search failed: {error}")

        if use_speech and transcript_path is not None:
            try:
                with st.spinner("Searching commentary..."):
                    transcript = json.loads(
                        transcript_path.read_text(encoding="utf-8")
                    )
                    records = transcript["segments"]

                    results["Spoken commentary"] = (
                        search_speech(records, query, index_root)
                        if records else []
                    )

            except Exception as error:
                st.error(f"Speech search failed: {error}")

    st.session_state[results_key] = {
        "query": query,
        "groups": results,
    }

saved = st.session_state.get(results_key)

if saved and saved["query"]:
    st.write(f"Results for: **{saved['query']}**")
    st.caption(
        "Closest matches may still be irrelevant. "
        "Similarity scores are not confidence percentages "
        "and should not be compared across search types."
    )

    for source, matches in saved["groups"].items():
        st.subheader(source)

        if not matches:
            st.info("No searchable content available.")

        for i, match in enumerate(matches):
            with st.container(border=True):
                if match["frame"] and Path(match["frame"]).exists():
                    st.image(match["frame"], width=280)

                st.write(f"**{label(match['time'])}**")

                if match["text"]:
                    st.write(match["text"])

                st.caption(f"Similarity: {match['score']:.3f}")

                if st.button(
                    "Jump here",
                    key=f"{results_key}_{source}_{i}",
                ):
                    st.session_state[play_key] = max(
                        0, int(match["time"]) - 1
                    )

player.video(
    str(video),
    start_time=st.session_state.get(play_key, 0),
)