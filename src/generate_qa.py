import os
import json
import csv
import time
import glob
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).parent.parent / ".env")

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
OUTPUT_DIR = Path(__file__).parent.parent / "training_data"
JSONL_PATH = OUTPUT_DIR / "cybersecurity_qa.jsonl"
CSV_PATH = OUTPUT_DIR / "cybersecurity_qa_review.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def _extract_pairs_from_raw(raw: str) -> list[dict]:
    """Parse QA pairs from LLM response, repairing truncated JSON when needed."""
    # strip markdown code fences
    if "```" in raw:
        parts = raw.split("```")
        # take the first fenced block
        for i, part in enumerate(parts):
            if i % 2 == 1:  # odd indices are inside fences
                raw = part.lstrip("json").strip()
                break

    start = raw.find("[")
    if start == -1:
        raise ValueError("No JSON array start '[' found")

    # try the raw slice first, then try repair suffixes for truncated responses
    slice_ = raw[start:]
    for suffix in ("", "}]", "\"}]", "\"}\n]"):
        candidate = slice_ + suffix if suffix else slice_
        # find last complete ']' position
        end = candidate.rfind("]") + 1
        if end == 0:
            continue
        try:
            pairs = json.loads(candidate[:end])
            return [p for p in pairs if "question" in p and "answer" in p]
        except json.JSONDecodeError:
            continue

    # last resort: extract individual objects with regex
    import re
    objects = re.findall(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', raw, re.DOTALL)
    pairs = []
    for obj in objects:
        try:
            pairs.append(json.loads(obj))
        except json.JSONDecodeError:
            pass
    if pairs:
        return pairs

    raise ValueError(f"Could not parse response:\n{raw[:400]}")


def generate_qa_pairs(doc_text: str, num_pairs: int = 6) -> list[dict]:
    for attempt in range(3):
        effective = max(3, num_pairs - attempt)  # reduce pairs on retry
        prompt = (
            f"You are creating a cybersecurity training dataset. "
            f"Read the document below and generate exactly {effective} question-answer pairs.\n\n"
            "Rules:\n"
            "- Vary question types: factual, conceptual, 'how does', 'what is the difference', 'why'\n"
            "- Every answer must be grounded strictly in the document — no outside knowledge\n"
            "- Answers: 2-3 sentences, specific and informative\n"
            "- Questions must be self-contained (no 'the above', 'this document', etc.)\n"
            "- Return ONLY a valid JSON array, no markdown fences, no extra text\n\n"
            f"Document:\n{doc_text}\n\n"
            "Return ONLY this JSON (no other text):\n"
            '[{"question": "...", "answer": "..."}, ...]'
        )
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            pairs = _extract_pairs_from_raw(raw)
            if pairs:
                return pairs
        except ValueError:
            if attempt == 2:
                raise
            time.sleep(2)
    return []


def load_checkpoint() -> set:
    if CHECKPOINT_PATH.exists():
        return set(json.loads(CHECKPOINT_PATH.read_text()))
    return set()


def save_checkpoint(done: set):
    CHECKPOINT_PATH.write_text(json.dumps(sorted(done)))


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    corpus_files = sorted(glob.glob(str(CORPUS_DIR / "*.txt")))
    if not corpus_files:
        raise FileNotFoundError(f"No .txt files in {CORPUS_DIR}")

    done = load_checkpoint()
    print(f"Found {len(corpus_files)} documents. {len(done)} already processed.")

    # open files in append mode so we can resume
    jsonl_file = open(JSONL_PATH, "a", encoding="utf-8")
    csv_rows: list[dict] = []

    # read existing CSV rows if resuming
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

    total_pairs = sum(1 for _ in open(JSONL_PATH, encoding="utf-8")) if JSONL_PATH.exists() else 0
    print(f"Starting from {total_pairs} existing pairs.\n")

    try:
        for idx, filepath in enumerate(corpus_files, 1):
            doc_name = Path(filepath).name
            if doc_name in done:
                print(f"[{idx:02d}/{len(corpus_files)}] SKIP  {doc_name}")
                continue

            doc_text = Path(filepath).read_text(encoding="utf-8").strip()
            word_count = len(doc_text.split())

            # aim for more pairs on longer docs
            num_pairs = 7 if word_count > 600 else 5

            print(f"[{idx:02d}/{len(corpus_files)}] Generating {num_pairs} pairs for {doc_name} ({word_count}w) ...", end="", flush=True)

            try:
                pairs = generate_qa_pairs(doc_text, num_pairs)
            except Exception as e:
                print(f" ERROR: {e}")
                time.sleep(3)
                continue

            for pair in pairs:
                question = pair["question"].strip()
                answer = pair["answer"].strip()
                context = doc_text

                # write JSONL line
                record = {
                    "prompt": f"Question: {question}\nContext: {context}\nAnswer:",
                    "completion": f" {answer}",
                }
                jsonl_file.write(json.dumps(record) + "\n")

                # collect CSV row
                csv_rows.append({
                    "source_doc": doc_name,
                    "question": question,
                    "answer": answer,
                })

            jsonl_file.flush()
            total_pairs += len(pairs)
            done.add(doc_name)
            save_checkpoint(done)
            print(f" done ({len(pairs)} pairs, total={total_pairs})")

            # respect Groq rate limit: ~30 req/min free tier
            if idx < len(corpus_files):
                time.sleep(2.2)

    finally:
        jsonl_file.close()

    # write CSV (overwrite with all rows)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_doc", "question", "answer"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # clean up checkpoint on success
    if len(done) == len(corpus_files):
        CHECKPOINT_PATH.unlink(missing_ok=True)
        print(f"\nDone! {total_pairs} pairs written to:")
        print(f"  {JSONL_PATH}")
        print(f"  {CSV_PATH}")
    else:
        remaining = len(corpus_files) - len(done)
        print(f"\nPartial run: {total_pairs} pairs so far, {remaining} docs remaining.")
        print("Re-run the script to continue from the checkpoint.")


if __name__ == "__main__":
    main()
