import os
import json
import re

TXT_DIR = "txts"
JSON_DIR = "json"
OUTPUT_FILE = "training_data.json"

def extract_number(text):
    match = re.search(r"(\d+)\)", text)
    return int(match.group(1)) if match else None

def clean_answer(ans):
    return ans.replace("ANSWER:", "").strip()

def build_structured_from_json(path):
    with open(path) as f:
        data = json.load(f)

    results = []
    current = {}

    for item in data:
        label = item["label"]
        text = item["text"].strip()

        if label == "TOSSUP_TEXT":
            current = {}
            current["number"] = extract_number(text)
            current["tossup"] = text

        elif label == "TOSSUP_ANSWER":
            current["tossup_answer"] = clean_answer(text)

        elif label == "BONUS_TEXT":
            current["bonus"] = text

        elif label == "BONUS_ANSWER":
            current["bonus_answer"] = clean_answer(text)

            if all(k in current for k in ["number", "tossup", "tossup_answer", "bonus", "bonus_answer"]):
                results.append(current)
                current = {}

    return results

def split_raw_chunks(text):
    parts = re.split(r"(TOSS-UP)", text)

    chunks = []
    buffer = ""

    for part in parts:
        if part == "TOSS-UP":
            if buffer:
                chunks.append(buffer)
            buffer = part
        else:
            buffer += part

    if buffer:
        chunks.append(buffer)

    return chunks

def extract_number_from_chunk(chunk):
    match = re.search(r"(\d+)\)", chunk)
    return int(match.group(1)) if match else None

all_examples = []

for file in os.listdir(TXT_DIR):
    if not file.endswith(".txt"):
        continue

    base = file.replace(".txt", "")
    txt_path = os.path.join(TXT_DIR, file)
    json_path = os.path.join(JSON_DIR, base + ".json")

    if not os.path.exists(json_path):
        print(f"Skipping {file} (no matching JSON)")
        continue

    print(f"Processing {file}...")

    # Load raw text
    with open(txt_path) as f:
        raw_text = f.read()

    # Split raw chunks
    raw_chunks = split_raw_chunks(raw_text)

    # Build structured targets
    structured = build_structured_from_json(json_path)

    # Index structured by number
    structured_by_num = {q["number"]: q for q in structured if q["number"] is not None}

    # Match chunks to structured data
    for chunk in raw_chunks:
        num = extract_number_from_chunk(chunk)

        if num is None:
            continue

        if num not in structured_by_num:
            continue

        target = structured_by_num[num]

        example = {
            "input": "extract_science_bowl:\n" + chunk.strip(),
            "output": json.dumps(target)
        }

        all_examples.append(example)

print(f"\n #examples: {len(all_examples)}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_examples, f, indent=2)

print(f"saved to {OUTPUT_FILE}")