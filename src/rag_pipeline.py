"""
FYP Baseline RAG Pipeline — CB013309
======================================
Pipeline architecture (updated):
  - Corpus:     all .txt files in corpus/ (no limit)
  - Chunking:   SemanticChunker (all-MiniLM-L6-v2) as child splitter,
                ParentDocumentRetriever stores full source docs as parents
  - Retrieval:  k=10 candidate parent docs via child-level semantic search
  - Reranking:  cross-encoder/ms-marco-MiniLM-L-6-v2 → top 3 passed to LLM
  - LLM:        openai/gpt-oss-20b via Groq
  - Scoring:    cosine-similarity hallucination confidence per sentence

Next phases:
  - Fine-tune a small model on domain Q&A pairs (Step 5)
  - Swap placeholder fine-tuned model into eval dashboard (Step 6)
  - RAGAS evaluation on both pipelines (Step 7)
"""

import os
import re
from typing import Any, List
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import CrossEncoder
from pydantic import Field

CORPUS_DIR   = os.path.join(os.path.dirname(__file__), "..", "corpus")
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lora_adapter")
PHI2_BASE    = "microsoft/phi-2"
INITIAL_K    = 10   # candidates retrieved before reranking
TOP_K        = 3    # docs passed to LLM after reranking

_ft_model_cache: dict = {}


# ── Custom retrievers ─────────────────────────────────────────────────────────
class ParentChildRetriever(BaseRetriever):
    """Search child (semantic) chunks; return the corresponding full parent doc."""

    vectorstore:  Any = Field(...)
    parent_store: Any = Field(...)   # dict[source_name -> Document]
    initial_k:    int = INITIAL_K

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        child_hits = self.vectorstore.similarity_search(query, k=self.initial_k)
        seen, parents = set(), []
        for chunk in child_hits:
            src = chunk.metadata.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                parents.append(self.parent_store.get(src, chunk))
        return parents


class RerankedRetriever(BaseRetriever):
    """Rerank candidates from a base retriever with a cross-encoder, return top-k."""

    base_retriever: Any = Field(...)
    cross_encoder:  Any = Field(...)
    final_k:        int = TOP_K

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        candidates = self.base_retriever.invoke(query)
        if not candidates:
            return []
        pairs  = [[query, doc.page_content] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:self.final_k]]


# ── Document loading ──────────────────────────────────────────────────────────
def load_corpus():
    import glob
    docs      = []
    txt_files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in '{CORPUS_DIR}/'. Add your domain documents there."
        )
    for path in txt_files:
        loader = TextLoader(path, encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = os.path.basename(path)
        docs.extend(loaded)
    print(f"Loaded {len(docs)} document(s) from '{CORPUS_DIR}/'")
    return docs


# ── Retriever: semantic chunking + parent-child + cross-encoder reranking ──────
def build_retriever(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Semantic child chunks — variable-size, meaning-preserving splits
    print("Indexing with semantic chunking (first run may take a moment)…")
    child_chunks = SemanticChunker(embeddings).split_documents(docs)
    print(f"Created {len(child_chunks)} semantic child chunks from {len(docs)} document(s).")

    # Parent store: one entry per source file (full document text)
    parent_store = {}
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in parent_store:
            parent_store[src] = doc

    # Vectorstore is indexed on child chunks for precision; retrieval returns parents
    vectorstore = Chroma.from_documents(child_chunks, embeddings)

    base_retriever = ParentChildRetriever(
        vectorstore=vectorstore,
        parent_store=parent_store,
    )

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    retriever = RerankedRetriever(
        base_retriever=base_retriever,
        cross_encoder=cross_encoder,
    )
    print(f"Ready — {INITIAL_K} candidates → rerank → top {TOP_K} to LLM.")
    return retriever, embeddings


# ── Hallucination scorer ──────────────────────────────────────────────────────
SUPPORT_THRESHOLD = 0.35  # cosine similarity below this = weakly supported

def score_hallucination(answer, source_docs, embeddings):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer)
                 if len(s.strip()) > 15]
    if not sentences:
        return 100.0, []

    context_texts = [doc.page_content for doc in source_docs]
    sent_vecs = np.array(embeddings.embed_documents(sentences))
    ctx_vecs  = np.array(embeddings.embed_documents(context_texts))
    sent_vecs /= np.linalg.norm(sent_vecs, axis=1, keepdims=True)
    ctx_vecs  /= np.linalg.norm(ctx_vecs,  axis=1, keepdims=True)
    sim_matrix = sent_vecs @ ctx_vecs.T
    max_sims   = sim_matrix.max(axis=1)
    results    = [(sent, float(sim), float(sim) >= SUPPORT_THRESHOLD)
                  for sent, sim in zip(sentences, max_sims)]
    confidence = float(np.mean(max_sims)) * 100
    return confidence, results


# ── RAG chain ─────────────────────────────────────────────────────────────────
def build_chain(retriever):
    prompt_template = """You are a cybersecurity domain expert.
Using the retrieved context below, give a clear, synthesised answer drawn from the available sources.
Always provide a concrete, best-effort answer — never refuse to answer.
If the context only partially covers the question, answer what you can and note the gap briefly.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    llm   = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain


# ── Shared query helper (used by CLI and app.py) ──────────────────────────────
def query_pipeline(chain, question, embeddings):
    """Run question through chain; return structured result dict."""
    result      = chain.invoke({"query": question})
    answer      = result["result"]
    source_docs = result["source_documents"]

    seen, sources = set(), []
    for doc in source_docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append((src, doc.page_content[:300]))

    confidence, sentence_scores = score_hallucination(answer, source_docs, embeddings)
    n_supported = sum(1 for _, _, ok in sentence_scores if ok)

    return {
        "answer":          answer,
        "sources":         sources,
        "source_docs":     source_docs,
        "confidence":      confidence,
        "sentence_scores": sentence_scores,
        "n_supported":     n_supported,
    }


# ── Fine-tuned model (Phi-2 + QLoRA) ─────────────────────────────────────────
def load_finetuned_model():
    """Load Phi-2 base + LoRA adapter once per session; return (model, tokenizer)."""
    if "model" in _ft_model_cache:
        return _ft_model_cache["model"], _ft_model_cache["tokenizer"]

    if not os.path.isdir(ADAPTER_PATH):
        raise FileNotFoundError(
            f"LoRA adapter not found at '{ADAPTER_PATH}'. "
            "Train the model first or place the adapter directory there."
        )

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"Loading {PHI2_BASE} in float16 on CPU…")
    tokenizer = AutoTokenizer.from_pretrained(PHI2_BASE, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        PHI2_BASE,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    base_model.to("cpu")

    print(f"Attaching LoRA adapter from '{ADAPTER_PATH}'…")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    _ft_model_cache["model"]     = model
    _ft_model_cache["tokenizer"] = tokenizer
    print("Fine-tuned model ready.")
    return model, tokenizer


PHI2_MAX_POSITIONS_FALLBACK = 2048  # used only if introspection below fails
CONTEXT_TRUNCATION_MARGIN = 50  # headroom for special tokens / retokenization drift
GENERATION_TOKEN_BUDGET = 256  # reserved for model.generate()'s max_new_tokens


def _real_max_positions(model, tokenizer) -> int:
    """Phi-2's actual context window, read from the loaded model/tokenizer rather
    than assumed, so this never silently drifts from whatever checkpoint is loaded."""
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if not max_positions:
        max_positions = getattr(tokenizer, "model_max_length", None)
    if not max_positions or max_positions > 1_000_000:  # unset sentinel guard
        max_positions = PHI2_MAX_POSITIONS_FALLBACK
    return max_positions


def query_finetuned(question: str, retriever, embeddings) -> dict:
    """Run question through fine-tuned Phi-2+LoRA; return same dict shape as query_pipeline()."""
    import torch

    model, tokenizer = load_finetuned_model()

    source_docs = retriever.invoke(question)
    context     = "\n\n".join(doc.page_content for doc in source_docs)

    header = (
        "You are a cybersecurity domain expert.\n"
        "Using the retrieved context below, give a clear, synthesised answer drawn from the available sources.\n"
        "Always provide a concrete, best-effort answer — never refuse to answer.\n"
        "If the context only partially covers the question, answer what you can and note the gap briefly.\n\n"
        "Context:\n"
    )
    footer = f"\n\nQuestion: {question}\n\nAnswer:"

    # Reserve space for the fixed instruction + Question/Answer template first,
    # so truncation (if needed) only ever eats into the context — never the
    # question, which previously got silently cut off by right-side truncation
    # on the full assembled prompt. Also reserve GENERATION_TOKEN_BUDGET so the
    # prompt + generated tokens together stay within the model's real context window.
    real_max_positions = _real_max_positions(model, tokenizer)

    template_token_count = len(tokenizer(header + footer)["input_ids"])
    available_context_tokens = max(
        0,
        real_max_positions
        - template_token_count
        - GENERATION_TOKEN_BUDGET
        - CONTEXT_TRUNCATION_MARGIN,
    )

    # verbose=False: this intentionally tokenizes the raw, untruncated context to
    # measure how much would need trimming — it's expected to exceed the model's
    # max length for long retrievals, so the library's generic overflow warning
    # would be misleading noise here. We log our own warning below instead, only
    # when truncation actually happens.
    original_context_tokens = len(tokenizer(context, verbose=False)["input_ids"])

    prior_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        truncated_context_ids = tokenizer(
            context, truncation=True, max_length=available_context_tokens
        )["input_ids"]
    finally:
        tokenizer.truncation_side = prior_truncation_side

    used_context_tokens = len(truncated_context_ids)
    context_truncated = used_context_tokens < original_context_tokens
    trimmed_context = tokenizer.decode(truncated_context_ids, skip_special_tokens=True)

    if context_truncated:
        print(
            f"WARNING: context truncated to fit the token budget — would have been "
            f"{template_token_count + original_context_tokens} tokens (over the "
            f"{real_max_positions}-token limit); trimmed context from "
            f"{original_context_tokens} to {used_context_tokens} tokens."
        )

    prompt = header + trimmed_context + footer

    # Hard safety net: retokenizing the reassembled string can drift a few tokens
    # from the token-id-level budget above (BPE merges differ at the splice
    # points). If that ever pushes the prompt + generation budget over the
    # model's real limit, trim the context a little further and retry, rather
    # than silently handing the model an over-length sequence.
    for _ in range(3):
        inputs = tokenizer(prompt, return_tensors="pt", verbose=False)
        final_prompt_tokens = inputs["input_ids"].shape[-1]
        if final_prompt_tokens + GENERATION_TOKEN_BUDGET <= real_max_positions:
            break
        overflow = final_prompt_tokens + GENERATION_TOKEN_BUDGET - real_max_positions
        print(
            f"WARNING: reassembled prompt drifted {overflow} token(s) over budget "
            f"({final_prompt_tokens} prompt + {GENERATION_TOKEN_BUDGET} generation > "
            f"{real_max_positions}) — trimming context further before generation."
        )
        tokenizer.truncation_side = "left"
        truncated_context_ids = tokenizer(
            trimmed_context, truncation=True,
            max_length=max(0, len(truncated_context_ids) - overflow - 5),
        )["input_ids"]
        tokenizer.truncation_side = prior_truncation_side
        used_context_tokens = len(truncated_context_ids)
        context_truncated = True
        trimmed_context = tokenizer.decode(truncated_context_ids, skip_special_tokens=True)
        prompt = header + trimmed_context + footer
    else:
        final_prompt_tokens = tokenizer(prompt, return_tensors="pt", verbose=False)["input_ids"].shape[-1]

    assert final_prompt_tokens + GENERATION_TOKEN_BUDGET <= real_max_positions, (
        f"Prompt ({final_prompt_tokens} tokens) + generation budget "
        f"({GENERATION_TOKEN_BUDGET}) still exceeds the model's real context window "
        f"({real_max_positions}) after correction — this should never happen."
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=GENERATION_TOKEN_BUDGET,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer     = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    seen, sources = set(), []
    for doc in source_docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append((src, doc.page_content[:300]))

    confidence, sentence_scores = score_hallucination(answer, source_docs, embeddings)
    n_supported = sum(1 for _, _, ok in sentence_scores if ok)

    return {
        "answer":          answer,
        "sources":         sources,
        "source_docs":     source_docs,
        "confidence":      confidence,
        "sentence_scores": sentence_scores,
        "n_supported":     n_supported,
        "context_truncated":       context_truncated,
        "original_context_tokens": original_context_tokens,
        "used_context_tokens":     used_context_tokens,
    }


# ── CLI display ───────────────────────────────────────────────────────────────
def run_query(chain, question, embeddings):
    r = query_pipeline(chain, question, embeddings)
    print("\n" + "=" * 60)
    print(f"QUESTION:\n  {question}\n")
    print(f"ANSWER:\n  {r['answer']}\n")
    print(f"CONFIDENCE: {r['confidence']:.0f}%  |  "
          f"Supported: {r['n_supported']}/{len(r['sentence_scores'])} sentences")
    for sent, sim, supported in r["sentence_scores"]:
        tag = "[+]" if supported else "[-]"
        print(f"  {tag} ({sim:.2f})  {sent}")
    print(f"\nSOURCES ({len(r['sources'])} document(s) cited):")
    for i, (src, excerpt) in enumerate(r["sources"], 1):
        print(f"  [{i}] {src}")
        print(f"       \"{excerpt.strip()[:150]}...\"")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("FYP Baseline RAG — CB013309")
    print("Cybersecurity | openai/gpt-oss-20b (Groq) | Semantic chunking + cross-encoder reranking\n")

    docs                  = load_corpus()
    retriever, embeddings = build_retriever(docs)
    chain                 = build_chain(retriever)

    print("\nSystem ready. Type a question or 'quit' to exit.\n")
    while True:
        question = input("Question:\n> ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        run_query(chain, question, embeddings)


if __name__ == "__main__":
    main()
