import importlib
import time

modules = [
    "streamlit",
    "torch",
    "sentence_transformers",
    "cv2",
    "easyocr",
    "faster_whisper",
]

total_start = time.perf_counter()

for module_name in modules:
    print(f"Loading {module_name}...", flush=True)
    start = time.perf_counter()

    try:
        importlib.import_module(module_name)
    except Exception as error:
        print(f"  FAILED: {error}", flush=True)
    else:
        elapsed = time.perf_counter() - start
        print(f"  {elapsed:.2f} seconds", flush=True)

total = time.perf_counter() - total_start
print(f"\nTotal import time: {total:.2f} seconds")