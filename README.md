# Cybersecurity RAG Q&A System — Final Year Project

## What this is

This project is a domain-specific question-answering system for cybersecurity topics, built on
retrieval-augmented generation (RAG). It compares two answer-generation approaches sharing the
same retrieval pipeline: a general-purpose baseline model (`openai/gpt-oss-20b`, accessed via the
Groq API) and a QLoRA fine-tuned Phi-2 model trained specifically on cybersecurity Q&A data. Both
pipelines retrieve relevant passages from a cybersecurity document corpus (semantic chunking plus
cross-encoder reranking), generate an answer grounded in that context, and score the answer for
hallucination risk (how well each sentence is supported by the retrieved sources).

Beyond just comparing the two models' answers, the project specifically evaluates **confidence
calibration** — whether each model's self-reported confidence score actually tracks whether its
answers are correct, using a set of human-labeled evaluation questions. This distinguishes raw
accuracy from trustworthiness: a model can be less accurate overall yet still be more useful in
practice if its confidence reliably signals when it doesn't know something.

## Why this runs on Google Colab rather than locally

Running the fine-tuned Phi-2 model requires GPU acceleration to be practical — on CPU-only
hardware, generating a single answer can take tens of minutes, and BitsAndBytes 4-bit
quantization requires CUDA, which isn't available on Apple Silicon (MPS). Since local development
hardware here doesn't have a suitable GPU, all fine-tuned-model work (training, evaluation, and
the live demo) is run on Google Colab's free-tier T4 GPU instead. The Streamlit demo app is
launched inside a Colab notebook and tunnelled out to a normal browser URL via `ngrok`, so it can
still be used and demoed like any other web app, just hosted on Colab's GPU rather than a local
machine.

## What's in this repo

```
fyp_v2/
├── corpus/                              Source cybersecurity documents (OWASP, NIST, SANS)
│                                         used to build the retrieval index.
├── training_data/
│   ├── cybersecurity_qa.jsonl            Generated QA dataset used for fine-tuning + evaluation.
│   └── cybersecurity_qa_review.csv       Manual review pass over the generated QA pairs.
├── src/
│   ├── rag_pipeline.py                  Core retrieval + generation pipeline (both baseline and
│   │                                    fine-tuned paths): corpus loading, semantic chunking,
│   │                                    cross-encoder reranking, generation, hallucination scoring.
│   ├── app.py                           Streamlit dashboard — live Q&A, baseline-vs-fine-tuned
│   │                                    comparison, and the calibration analysis view.
│   └── generate_qa.py                   Generates the QA dataset from the corpus documents.
├── notebooks/
│   ├── finetune_cybersecurity.ipynb      Fine-tunes Phi-2 with QLoRA on the cybersecurity QA
│   │                                    dataset.
│   ├── finetune_inference_eval.ipynb    Runs the held-out evaluation questions through the
│   │                                    fine-tuned model on a Colab GPU.
│   └── streamlit_demo_colab.ipynb       Launches the Streamlit demo app on a Colab GPU and
│                                        tunnels it out via ngrok for browser access.
├── run_evaluation.py                    Runs the baseline model evaluation over the held-out
│                                        question set.
├── evaluation_results.csv               Baseline + fine-tuned evaluation output (wide format).
├── Finetuned Eval Results Long.csv       Fine-tuned evaluation results from the Colab GPU run
│                                        (long format).
├── Calibration Dataset for Model Analysis.csv   Human-labeled correctness data powering the
│                                        Calibration tab in the Streamlit app.
├── models/lora_adapter/                 Trained LoRA adapter (config + weights + tokenizer).
├── requirements.txt                     Python dependencies.
└── .gitignore
```

## Running it

1. Install dependencies: `pip install -r requirements.txt`
2. Create a `.env` file at the repo root with your own Groq API key:
   `GROQ_API_KEY=your-key-here` (never commit this file — it's already in `.gitignore`).
3. From there, two options:
   - **Baseline-only, locally**: run `streamlit run src/app.py`. This works fully on CPU, but
     the fine-tuned model won't be available unless you also have a CUDA GPU locally.
   - **Full system (fine-tuned model + demo), on Colab**: open `notebooks/streamlit_demo_colab.ipynb`
     in Google Colab (GPU runtime), follow its steps to mount your data, install dependencies,
     and launch the demo with a public tunnel link. The notebook tunnels the demo out via
     `ngrok`, which requires a free ngrok.com account and an authtoken — it looks for the token
     in Colab's Secrets manager under the name `NGROK_AUTHTOKEN`, or prompts for it directly if
     not found there. Use `notebooks/finetune_inference_eval.ipynb` to reproduce the fine-tuned
     model's evaluation results, and `notebooks/finetune_cybersecurity.ipynb` to reproduce the
     fine-tuning process itself.
