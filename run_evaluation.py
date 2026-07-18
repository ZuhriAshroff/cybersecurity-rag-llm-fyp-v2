"""
FYP Evaluation Runner — CB013309
==================================
Runs the held-out test questions through BOTH the baseline pipeline
(query_pipeline) and the fine-tuned pipeline (query_finetuned), and
logs answers + confidence scores to a CSV for later analysis.

This script does NOT compute RAGAS scores — see score_ragas.py for
that, which reads the CSV this script produces. Keeping these two
steps separate means a RAGAS/dependency issue never costs you a
re-run of inference (which burns Groq calls + GPU time).

Usage:
    python run_evaluation.py
    python run_evaluation.py --limit 5        # quick smoke test
    python run_evaluation.py --skip-finetuned # baseline only

Expects:
    - src/rag_pipeline.py importable (run from repo root, or adjust
      sys.path below)
    - training_data/cybersecurity_qa.jsonl with the held-out test
      split. By default this script uses the LAST 20% of the file
      (45 questions if the file has 225 total), matching the 80/20
      split mentioned in your training notes. If you saved a separate
      test-only file, point --qa-file at it directly and the script
      will use ALL questions in it.
    - models/lora_adapter/ present locally (for the fine-tuned run)
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback

# ── Path setup ─────────────────────────────────────────────────────────────
# This script lives at repo root (fyp_v2/run_evaluation.py), so REPO_ROOT
# is just the script's own directory — not one level up.
REPO_ROOT = os.environ.get("FYP_REPO_ROOT", os.path.dirname(os.path.abspath(__file__)))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

QA_FILE_DEFAULT = os.path.join(REPO_ROOT, "training_data", "cybersecurity_qa.jsonl")
OUTPUT_CSV_DEFAULT = os.path.join(REPO_ROOT, "evaluation_results.csv")

# Your cybersecurity_qa.jsonl uses {"prompt": "...", "completion": "..."} —
# the training format, not a clean question/answer pair. "prompt" embeds the
# question inside a larger template: "Question: <q>\nContext: <ctx>\nAnswer:"
# We extract just the question text below; "completion" is used as-is as the
# reference answer.
import re as _re
_QUESTION_PATTERN = _re.compile(r"Question:\s*(.+?)\s*\nContext:", _re.DOTALL)


def extract_question(row: dict) -> str:
    """Pull the bare question out of the training-format 'prompt' field.
    Falls back to a plain 'question' key if present (in case the file format
    changes later), and as a last resort uses the raw prompt text."""
    if "question" in row and row["question"]:
        return row["question"].strip()

    prompt_text = row.get("prompt", "")
    match = _QUESTION_PATTERN.search(prompt_text)
    if match:
        return match.group(1).strip()

    # Fallback: couldn't parse it out — return the raw prompt so it's at
    # least visible in the CSV for manual inspection, rather than silently
    # skipping the row.
    return prompt_text.strip()


def extract_reference_answer(row: dict) -> str:
    if "answer" in row and row["answer"]:
        return row["answer"].strip()
    return row.get("completion", "").strip()


def load_test_questions(qa_file: str, limit, test_split_only: bool):
    """Load questions from the QA jsonl. If test_split_only, take the last 20%
    (mirrors the 80/20 split used in training, so eval doesn't leak training rows)."""
    rows = []
    with open(qa_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if test_split_only:
        split_idx = int(len(rows) * 0.8)
        rows = rows[split_idx:]
        print(f"Using held-out test split: last {len(rows)} of {split_idx + len(rows)} total rows.")
    else:
        print(f"Using all {len(rows)} rows in '{qa_file}' as the test set.")

    if limit:
        rows = rows[:limit]
        print(f"Limiting to first {len(rows)} question(s) for this run.")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Run baseline + fine-tuned eval over held-out questions.")
    parser.add_argument("--qa-file", default=QA_FILE_DEFAULT,
                         help="Path to cybersecurity_qa.jsonl (or a dedicated test-only file).")
    parser.add_argument("--output", default=OUTPUT_CSV_DEFAULT,
                         help="Path to write the results CSV.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N questions (useful for a quick smoke test).")
    parser.add_argument("--skip-finetuned", action="store_true",
                         help="Only run the baseline pipeline (e.g. if the adapter isn't ready yet).")
    parser.add_argument("--full-file-is-test-set", action="store_true",
                         help="Treat --qa-file as already being the held-out test set "
                              "(use ALL rows) instead of taking the last 20%% of a combined file.")
    args = parser.parse_args()

    if not os.path.isfile(args.qa_file):
        print(f"ERROR: QA file not found at '{args.qa_file}'.")
        print("Pass --qa-file /path/to/your/file.jsonl if it lives somewhere else.")
        sys.exit(1)

    questions = load_test_questions(
        args.qa_file, args.limit, test_split_only=not args.full_file_is_test_set
    )
    if not questions:
        print("ERROR: No questions loaded — check the QA file path/format.")
        sys.exit(1)

    # ── Import pipeline (after sys.path is set up) ──────────────────────────
    print("\nImporting rag_pipeline...")
    import rag_pipeline as rp

    print("Loading corpus and building retriever (this re-embeds the corpus — "
          "may take a minute)...")
    docs = rp.load_corpus()
    retriever, embeddings = rp.build_retriever(docs)
    chain = rp.build_chain(retriever)

    finetuned_ready = False
    if not args.skip_finetuned:
        try:
            print("\nLoading fine-tuned model (Phi-2 + LoRA adapter)...")
            rp.load_finetuned_model()
            finetuned_ready = True
            print("Fine-tuned model loaded OK.")
        except FileNotFoundError as e:
            print(f"\nWARNING: fine-tuned model not available — {e}")
            print("Continuing with --skip-finetuned behaviour (baseline only).")
        except Exception:
            print("\nWARNING: fine-tuned model failed to load. Full traceback:")
            traceback.print_exc()
            print("Continuing with baseline only.")

    # ── Run eval loop ─────────────────────────────────────────────────────
    fieldnames = [
        "question_id", "question", "reference_answer",
        "baseline_answer", "baseline_confidence", "baseline_n_supported",
        "baseline_n_sentences", "baseline_sources", "baseline_latency_sec", "baseline_error",
        "finetuned_answer", "finetuned_confidence", "finetuned_n_supported",
        "finetuned_n_sentences", "finetuned_sources", "finetuned_latency_sec", "finetuned_error",
    ]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(questions, 1):
            question = extract_question(row)
            reference = extract_reference_answer(row)
            if not question:
                print(f"  [{i}/{len(questions)}] skipping row with empty question")
                continue

            print(f"\n[{i}/{len(questions)}] {question[:80]}")
            out_row = {
                "question_id": i,
                "question": question,
                "reference_answer": reference,
            }

            # Baseline
            try:
                t0 = time.time()
                r = rp.query_pipeline(chain, question, embeddings)
                elapsed = time.time() - t0
                out_row.update({
                    "baseline_answer": r["answer"],
                    "baseline_confidence": round(r["confidence"], 2),
                    "baseline_n_supported": r["n_supported"],
                    "baseline_n_sentences": len(r["sentence_scores"]),
                    "baseline_sources": "; ".join(src for src, _ in r["sources"]),
                    "baseline_latency_sec": round(elapsed, 2),
                    "baseline_error": "",
                })
                print(f"  baseline:   confidence={r['confidence']:.1f}%  "
                      f"({r['n_supported']}/{len(r['sentence_scores'])} sentences)  "
                      f"[{elapsed:.1f}s]")
            except Exception as e:
                out_row.update({
                    "baseline_answer": "", "baseline_confidence": "",
                    "baseline_n_supported": "", "baseline_n_sentences": "",
                    "baseline_sources": "", "baseline_latency_sec": "",
                    "baseline_error": str(e),
                })
                print(f"  baseline:   ERROR — {e}")

            # Fine-tuned
            if finetuned_ready:
                try:
                    t0 = time.time()
                    r = rp.query_finetuned(question, retriever, embeddings)
                    elapsed = time.time() - t0
                    out_row.update({
                        "finetuned_answer": r["answer"],
                        "finetuned_confidence": round(r["confidence"], 2),
                        "finetuned_n_supported": r["n_supported"],
                        "finetuned_n_sentences": len(r["sentence_scores"]),
                        "finetuned_sources": "; ".join(src for src, _ in r["sources"]),
                        "finetuned_latency_sec": round(elapsed, 2),
                        "finetuned_error": "",
                    })
                    print(f"  finetuned:  confidence={r['confidence']:.1f}%  "
                          f"({r['n_supported']}/{len(r['sentence_scores'])} sentences)  "
                          f"[{elapsed:.1f}s]")
                except Exception as e:
                    out_row.update({
                        "finetuned_answer": "", "finetuned_confidence": "",
                        "finetuned_n_supported": "", "finetuned_n_sentences": "",
                        "finetuned_sources": "", "finetuned_latency_sec": "",
                        "finetuned_error": str(e),
                    })
                    print(f"  finetuned:  ERROR — {e}")
            else:
                out_row.update({
                    "finetuned_answer": "", "finetuned_confidence": "",
                    "finetuned_n_supported": "", "finetuned_n_sentences": "",
                    "finetuned_sources": "", "finetuned_latency_sec": "",
                    "finetuned_error": "skipped (model not loaded)",
                })

            writer.writerow(out_row)
            f.flush()  # write progressively — don't lose everything on a crash mid-run

    print(f"\nDone. Results written to: {args.output}")
    print(f"Rows: {i}")
    if not finetuned_ready:
        print("\nNOTE: fine-tuned column is empty for this run. Re-run once the adapter "
              "is in place — no need to redo baseline-only rows if you keep this file "
              "and merge, or just re-run the whole thing fresh once both are ready.")


if __name__ == "__main__":
    main()