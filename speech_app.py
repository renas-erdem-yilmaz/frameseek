import hashlib
import json
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="FrameSeek Speech", layout="wide")


@st.cache_resource
def load_model(model_size):
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1,
    )


def timestamp(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


st.title("FrameSeek — Speech Search")
st.caption("Find words and phrases spoken in your video.")

upload = st.file_uploader("Upload an MP4", type=["mp4"])

if upload is None:
    st.stop()

model_size = st.selectbox(
    "Model",
    ["tiny", "base"],
    help="Start with tiny. Base may improve accuracy but takes longer.",
)

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

# Separate transcripts for each model and language setting.
settings = f"{model_size}_{language or 'auto'}"
transcript_path = index_dir / f"speech_{settings}.json"
play_key = f"speech_play_{video_id}"

if st.button("Transcribe speech", type="primary"):
    status = st.empty()

    try:
        with st.spinner("Loading model and transcribing..."):
            model = load_model(model_size)

            segments, info = model.transcribe(
                str(video_path),
                language=language,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            records = []

            # Transcription happens as we iterate over segments.
            for segment in segments:
                text = segment.text.strip()

                if text:
                    records.append({
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": text,
                    })

                status.info(
                    f"Transcribed through {timestamp(segment.end)}"
                )

            transcript_path.write_text(
                json.dumps(
                    {
                        "language": info.language,
                        "segments": records,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        status.success("Transcript saved.")

    except Exception as error:
        st.error(f"Transcription failed: {error}")

player = st.empty()

if transcript_path.exists():
    transcript = json.loads(
        transcript_path.read_text(encoding="utf-8")
    )
    records = transcript["segments"]

    st.caption(f"Transcript language: {transcript['language']}")

    query = st.text_input(
        "Search spoken words",
        placeholder="Enter a word or phrase from the commentary",
    ).strip()

    matches = [
        record
        for record in records
        if query.casefold() in record["text"].casefold()
    ] if query else []

    if query and not matches:
        st.info("No matching words found in the transcript.")

    for i, record in enumerate(matches):
        with st.container(border=True):
            st.write(
                f"**{timestamp(record['start'])}** — {record['text']}"
            )

            if st.button("Jump here", key=f"speech_result_{i}"):
                st.session_state[play_key] = max(
                    0, int(record["start"]) - 1
                )

    with st.expander("Read the full transcript"):
        if not records:
            st.write("No speech detected.")

        for record in records:
            st.write(
                f"{timestamp(record['start'])} — {record['text']}"
            )

else:
    st.info("Click Transcribe speech to create a searchable transcript.")

player.video(
    str(video_path),
    start_time=st.session_state.get(play_key, 0),
)