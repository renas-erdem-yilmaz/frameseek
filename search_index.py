import json
from pathlib import Path


def main():
    index_path = Path("indexes/test/text.json")

    if not index_path.exists():
        raise RuntimeError("Run index_text.py first.")

    records = json.loads(index_path.read_text(encoding="utf-8"))

    print("Search for a name or phrase. Type 'quit' to exit.")

    while True:
        query = input("\nSearch text: ").strip()

        if query.lower() == "quit":
            break

        if not query:
            continue

        matches = 0

        for record in records:
            matching_text = [
                item["text"]
                for item in record["detections"]
                if query.casefold() in item["text"].casefold()
            ]

            if matching_text:
                minutes, seconds = divmod(record["timestamp"], 60)

                print(
                    f"{minutes:02d}:{seconds:02d}"
                    f" | {' / '.join(matching_text)}"
                    f" | {record['frame']}"
                )

                matches += 1

        if matches == 0:
            print("No matching text detected.")


if __name__ == "__main__":
    main()