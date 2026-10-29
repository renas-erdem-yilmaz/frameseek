import hashlib
import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="FrameSeek Speech", layout="wide")

SEARCH_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


@st.cache_resource
def load_transcriber(size):
    from faster_whisper import WhisperModel

    return WhisperModel(
        size,
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1,
    )


@st.cache_resource
def load_search_model():
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    return SentenceTransformer(SEARCH_MODEL, device="cpu")


def label(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def semantic_search(records, query, index_dir):
    import numpy as np

    texts = [record["text"] for record in records]

    # Changed transcripts or models get a separate embedding file.
    signature = hashlib.sha256(
        json.dumps(
            {"model": SEARCH_MODEL, "texts": texts},
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    embedding_path = index_dir / f"semantic_{signature}.npy"
    model = load_search_model()

    if embedding_path.exists():
        embeddings = np.load(embedding_path, allow_pickle=False)
    else:
        embeddings = model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        temporary_path = embedding_path.with_suffix(".tmp")

        with temporary_path.open("wb") as output:
            np.save(output, embeddings)

        temporary_path.replace(embedding_path)

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    scores = embeddings @ query_embedding
    positions = np.argsort(scores)[::-1][:5]

    return [
        {
            **records[int(position)],
            "detail": f"Similarity: {scores[position]:.3f}",
        }
        for position in positions
    ]


st.title("FrameSeek — Speech Search")
st.caption("Find exact words or search the meaning of the commentary.")

upload = st.file_uploader(
    "Upload an MP4",
    type=["mp4"],
    key="speech_upload",
)

if upload is None:
    st.stop()

size = st.selectbox("Transcription model", ["tiny", "base"])

language_name = st.selectbox(
    "Spoken language",
    ["Turkish", "English", "Auto-detect"],
)

language = {
    "Turkish": "tr",
    "English": "en",
    "Auto-detect": None,
}[language_name]

video_bytes = upload.getvalue()
video_id = hashlib.sha256(video_bytes).hexdigest()

video_dir = Path("videos") / video_id
index_dir = Path("indexes") / video_id

video_dir.mkdir(parents=True, exist_ok=True)
index_dir.mkdir(parents=True, exist_ok=True)

video_path = video_dir / "video.mp4"

if not video_path.exists():
    video_path.write_bytes(video_bytes)

settings = f"{size}_{language or 'auto'}"
transcript_path = index_dir / f"speech_{settings}.json"
play_key = f"speech_play_{video_id}"

if st.button("Transcribe speech", type="primary"):
    status = st.empty()

    try:
        with st.spinner("Loading model and transcribing..."):
            model = load_transcriber(size)

            segments, info = model.transcribe(
                str(video_path),
                language=language,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            records = []

            for segment in segments:
                text = segment.text.strip()

                if text:
                    records.append({
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": text,
                    })

                status.info(
                    f"Transcribed through {label(segment.end)}"
                )

            payload = {
                "language": info.language,
                "segments": records,
            }

            temporary_path = transcript_path.with_suffix(".tmp")
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_path.replace(transcript_path)

        status.success("Transcript saved.")

    except Exception as error:
        st.error(f"Transcription failed: {error}")

player = st.empty()

if transcript_path.exists():
    transcript_bytes = transcript_path.read_bytes()
    transcript = json.loads(transcript_bytes)
    records = transcript["segments"]

    # Results belong to this exact transcript.
    transcript_id = hashlib.sha256(transcript_bytes).hexdigest()
    results_key = f"speech_results_{video_id}_{transcript_id}"

    st.caption(f"Transcript language: {transcript['language']}")

    with st.form("speech_search_form"):
        mode = st.radio(
            "Search mode",
            ["Exact words", "Search by meaning"],
            horizontal=True,
        )

        query = st.text_input(
            "Search",
            placeholder="A name, phrase, or description",
        ).strip()

        submitted = st.form_submit_button("Search")

    if submitted:
        matches = []

        if query and records:
            if mode == "Exact words":
                matches = [
                    {**record, "detail": "Text match"}
                    for record in records
                    if query.casefold() in record["text"].casefold()
                ]
            else:
                try:
                    with st.spinner(
                        "Loading search model and comparing meanings..."
                    ):
                        matches = semantic_search(
                            records, query, index_dir
                        )
                except Exception as error:
                    st.error(f"Semantic search failed: {error}")

        st.session_state[results_key] = {
            "matches": matches,
            "mode": mode,
            "query": query,
        }

    saved = st.session_state.get(results_key)

    if saved and saved["query"]:
        st.write(f"Results for: **{saved['query']}**")

        if saved["mode"] == "Search by meaning":
            st.caption(
                "These are the closest transcript segments, "
                "even if none answers your query. "
                "Scores are not confidence percentages."
            )

        if not saved["matches"]:
            st.info("No matches found.")

        for i, record in enumerate(saved["matches"]):
            with st.container(border=True):
                st.write(
                    f"**{label(record['start'])}**"
                    f" — {record['text']}"
                )
                st.caption(record["detail"])

                if st.button(
                    "Jump here",
                    key=f"{results_key}_{i}",
                ):
                    st.session_state[play_key] = max(
                        0, int(record["start"]) - 1
                    )

    with st.expander("Read the full transcript"):
        if not records:
            st.write("No speech detected.")

        for record in records:
            st.write(
                f"{label(record['start'])} — {record['text']}"
            )

else:
    st.info("Click Transcribe speech to prepare a transcript.")

player.video(
    str(video_path),
    start_time=st.session_state.get(play_key, 0),
)