from pathlib import Path
import cv2

video_path = Path("videos/test.mp4")
output_folder = Path("frames/test")
interval_seconds = 2


def extract_frames():
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        video.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    output_folder.mkdir(parents=True, exist_ok=True)
    count = 0

    try:
        while True:
            timestamp = count * interval_seconds

            # OpenCV seeks using milliseconds.
            video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            success, frame = video.read()

            if not success:
                break

            filename = output_folder / f"frame_{timestamp:06d}s.jpg"

            if not cv2.imwrite(str(filename), frame):
                raise RuntimeError(f"Could not save: {filename}")

            print(f"Saved frame at {timestamp} seconds")
            count += 1
    finally:
        video.release()

    print(f"Done! Saved {count} frames to {output_folder}")


if __name__ == "__main__":
    extract_frames()