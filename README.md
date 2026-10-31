# FrameSeek

Search your videos using natural language, on-screen text, and spoken commentary. Find matching moments and jump to their timestamps in a local Streamlit app.

FrameSeek combines pretrained image–text, OCR, speech recognition, and text embedding models. It runs on the CPU and saves processed data locally for reuse.

## Features

- **Visual search:** describe a scene and retrieve similar video frames.
- **On-screen text search:** find names or phrases detected with OCR.
- **Speech transcription:** generate timestamped transcripts with Whisper.
- **Semantic speech search:** retrieve relevant commentary without matching its exact wording.
- **Combined search:** search visual scenes and speech from one page, with separate results for each source.
- **Timestamp playback:** preview frames and jump to matching moments.
- **Saved indexes:** reuse extracted embeddings and transcripts instead of processing the same video every time.

## How it works

For visual search, FrameSeek samples one frame every two seconds and uses CLIP to convert each frame into an embedding. It compares these embeddings with an embedding of your search description.

Optional OCR reads text appearing in sampled frames. Speech search uses Faster Whisper to transcribe the audio, then multilingual MiniLM to compare your query with transcript segments.

Each uploaded video is identified by a hash of its contents. Different videos receive separate folders for their frames, indexes, and transcripts.

## Models

| Task | Model or library |
| --- | --- |
| Image–text similarity | `sentence-transformers/clip-ViT-B-32` |
| On-screen text recognition | EasyOCR |
| Speech recognition | Faster Whisper, with multilingual `tiny` or `base` models |
| Semantic transcript search | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |

No training or fine-tuning is required. Model weights download on first use. Hosted inference API keys are not required by the app.

## Setup

Clone the repository:

```powershell
git clone https://github.com/renas-erdem-yilmaz/frameseek.git
cd frameseek
```

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install "sentence-transformers[image]" streamlit easyocr faster-whisper numpy pillow
```

EasyOCR installs `opencv-python-headless`, which provides the `cv2` module used for video processing. Avoid installing multiple OpenCV package variants in the same environment.

Start the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost
```

Open [localhost:8501](http://localhost:8501). Keep the terminal running while using the app. Press **Ctrl+C** to stop the server.

The commands above use Windows PowerShell. On macOS or Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

## Usage

### Visual and on-screen text search

1. Open **Visual & Text Search**.
2. Upload an MP4.
3. Enable **Read on-screen text** if you need OCR; leave it off for faster processing.
4. Click **Process video**.
5. Describe a scene in English, or search a literal name or phrase in text mode.
6. Click a result's timestamp button, then press play.

### Speech search

1. Open **Speech Search** and select the video.
2. Choose `tiny` or `base` and the spoken language.
3. Click **Transcribe speech**, or reuse a transcript saved with the same settings.
4. Choose **Exact words** or **Search by meaning**.
5. Search and jump to a matching transcript segment.

Semantic speech search supports multilingual matching, including English queries against Turkish commentary, though accuracy varies. Exact-word search matches the transcript's actual wording.

### Combined search

Open **Combined Search** and choose a saved video. Select a transcript if available, enter a query, and choose which sources to search. Visual results and commentary results appear separately.

This page searches existing indexes; prepare missing visual indexes or transcripts on their respective pages first. Saved videos currently appear under abbreviated content IDs and file sizes.

## App files

| File | Purpose |
| --- | --- |
| `app.py` | Main entry point and page navigation |
| `visual_app.py` | Video upload, frame processing, visual search, and OCR |
| `speech_app.py` | Transcription and exact or semantic speech search |
| `unified_search.py` | Combined search over saved videos |
| `videos/` | Local video copies |
| `frames/` | Extracted frame thumbnails |
| `indexes/` | Embeddings, OCR results, transcripts, and processing metadata |

The repository may also contain standalone scripts from the initial prototype. Start the current app with `app.py`.

## Performance

- The current implementation explicitly uses the **CPU**. An NVIDIA GPU is not required.
- Opening an idle page defers heavy AI imports where implemented. The first processing or search action can still take time to import libraries, download weights, and load models.
- Models are cached in memory during the running Streamlit process. Restarting the server clears that memory cache; completed indexes and transcripts remain on disk.
- OCR can be expensive. Start with a short clip and leave OCR disabled unless needed.
- The optimized visual pipeline batches embeddings and uses smaller thumbnails. Reduced OCR detection resolution can miss small text.
- Speech transcription runs separately from visual processing. The `tiny` model is a lighter starting point; `base` can improve transcription at a higher processing cost.

## Limitations

FrameSeek is an experimental retrieval prototype. It does not guarantee that a returned moment satisfies your query.

- **Similarity is not confidence.** A score such as `0.27` does not mean a 27% probability of correctness. Scores from different models should not be compared directly.
- **Closest matches can be irrelevant.** Semantic searches return the nearest candidates even when the requested content is absent.
- **Visual search sees sampled images.** It does not reliably understand motion, event order, or a sequence such as someone's first kill in a game.
- **Short events can be missed** between the two-second frame samples.
- **OCR and transcription can misread names**, subtitles, and small text.
- **Commentary is evidence of what was said.** It does not verify that the corresponding event happened on screen.
- Combined search presents separate source results; it does not perform event verification or a calibrated fusion of scores.
- Video playback depends on browser support for the uploaded file's codecs.

For example, searching for a scoped sniper scene is within the visual prototype's intended scope. Reliably identifying a specific player's first kill requires additional event detection and validation.

## Local data

Uploaded videos and generated indexes are stored on the machine running the app. In the local setup above, that is your computer. The app does not use a hosted inference service for processing, although initial model downloads require internet access.

Streamlit may collect usage telemetry according to its settings. To opt out, add the following to `.streamlit/config.toml`:

```toml
[browser]
gatherUsageStats = false
```

Keep personal media and generated data out of Git. The `.gitignore` should include:

```gitignore
.venv/
__pycache__/
videos/
frames/
indexes/
.env
```

## Status

Prototype covering video indexing, visual retrieval, OCR, speech transcription, semantic transcript retrieval, and timestamp playback. Search quality and processing time depend on the video, model, and hardware.
