#!/usr/bin/env python3
"""Tiny RAG mini-benchmark for movahedi.ca (Phase 4).

Corpus: public regulatory text (TBS Directive on Automated Decision-Making,
publications.gc.ca; OSFI Guideline E-23 summary via BLG, Nov 2025).
Retriever: sklearn TF-IDF, cosine similarity.
Generator: Gemini free tier via the gemini skill CLI (gemini-2.5-flash).
Judges: Ragas 0.4.3 metrics (faithfulness, context_precision_with_reference,
answer_relevancy) with a custom instructor-style LLM wrapper around the same
Gemini CLI and a TF-IDF embedding wrapper (local, deterministic).

Two variants:
  baseline: char chunk 400, overlap 0, top_k=3
  improved: char chunk 900, overlap 150, top_k=6
"""
import asyncio
import concurrent.futures
import json
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

WORK = Path.home() / "workspace/rai-portfolio/work"
GEMINI = Path.home() / "workspace/skills/gemini/bin/gemini_generate.py"
MODEL = "gemini-3.5-flash-lite"

# ---------------------------------------------------------------- gemini
_lock = threading.Lock()
_last_call = [0.0]
MIN_INTERVAL = 6.0  # stay under free-tier RPM


def call_gemini(prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        with _lock:
            wait = MIN_INTERVAL - (time.time() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.time()
        try:
            r = subprocess.run(
                [str(GEMINI), "--model", MODEL],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            time.sleep(10 * (attempt + 1))
            continue
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        if r.returncode == 0 and out:
            return out
        blob = (err + " " + out).lower()
        if "429" in blob or "rate" in blob or "quota" in blob or "resource_exhausted" in blob:
            time.sleep(20 * (attempt + 1))
            continue
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gemini call failed after {max_retries} retries")


# ---------------------------------------------------------------- chunking / retrieval
def chunk_text(text: str, size: int, overlap: int):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def build_chunks(passages, size, overlap):
    chunks = []
    for p in passages:
        for c in chunk_text(p["text"], size, overlap):
            chunks.append(f"[{p['id']} {p['title']}] {c}")
    return chunks


from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402


class TfidfRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english")
        self.mat = self.vec.fit_transform(chunks)

    def retrieve(self, query, k):
        q = self.vec.transform([query])
        scores = (self.mat @ q.T).toarray().ravel()
        idx = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in idx if scores[i] > 0] or [self.chunks[i] for i in idx]


GEN_SYSTEM = (
    "You are a careful assistant answering questions about Canadian AI regulation. "
    "Use ONLY the context passages provided. Answer in 2-3 sentences."
)


def generate_answer(question: str, contexts) -> str:
    ctx = "\n\n".join(f"Passage {i + 1}: {c}" for i, c in enumerate(contexts))
    prompt = (
        f"{GEN_SYSTEM}\n\nContext passages:\n{ctx}\n\nQuestion: {question}\n\n"
        "Rules: answer using only facts stated in the passages above. "
        'If the passages do not contain the answer, say exactly: '
        '"The provided documents do not contain this information." '
        "Do not add facts from outside the passages.\nAnswer:"
    )
    return call_gemini(prompt)


# ---------------------------------------------------------------- ragas glue
from ragas.llms.base import InstructorBaseRagasLLM  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbedding  # noqa: E402


class GeminiInstructorLLM(InstructorBaseRagasLLM):
    """Instructor-style LLM judge backed by the gemini skill CLI."""

    def _gen(self, prompt: str, response_model):
        schema = json.dumps(response_model.model_json_schema(), indent=1)
        full = (
            prompt
            + "\n\nReturn ONLY a JSON object (no markdown fences, no commentary) "
            + f"matching this JSON schema:\n{schema}\nThe response must be valid JSON."
        )
        last_err = ""
        for attempt in range(4):
            raw = call_gemini(full if attempt == 0 else full + f"\n\nYour previous output was invalid ({last_err}). Fix it and return only valid JSON.")
            try:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                obj = json.loads(m.group(0) if m else raw)
                return response_model.model_validate(obj)
            except Exception as e:  # noqa: BLE001
                last_err = str(e)[:200]
        raise RuntimeError(f"could not get valid JSON for {response_model.__name__}: {last_err}")

    def generate(self, prompt: str, response_model):
        return self._gen(prompt, response_model)

    async def agenerate(self, prompt: str, response_model):
        return await asyncio.to_thread(self._gen, prompt, response_model)


class TfidfEmbeddings(BaseRagasEmbedding):
    """Deterministic local TF-IDF embeddings for the answer_relevancy metric."""

    def __init__(self):
        super().__init__()
        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english", norm="l2")
        self._fitted = False

    def prefit(self, texts):
        self.vec.fit(texts)
        self._fitted = True

    def _ensure(self, texts):
        if not self._fitted:
            self.vec.fit(texts)
            self._fitted = True

    def embed_text(self, text, **kwargs):
        self._ensure([text])
        return self.vec.transform([text]).toarray()[0].tolist()

    async def aembed_text(self, text, **kwargs):
        return await asyncio.to_thread(self.embed_text, text)


# ---------------------------------------------------------------- main
VARIANTS = {
    "baseline": {"chunk_size": 400, "overlap": 0, "top_k": 3},
    "improved": {"chunk_size": 900, "overlap": 150, "top_k": 6},
}


def main():
    corpus = json.loads((WORK / "corpus.json").read_text())
    passages = corpus["passages"]
    questions = corpus["test_questions"]

    results = {}
    for vname, cfg in VARIANTS.items():
        print(f"[{datetime.now().isoformat()}] variant={vname}", flush=True)
        chunks = build_chunks(passages, cfg["chunk_size"], cfg["overlap"])
        retr = TfidfRetriever(chunks)
        samples = []
        for q in questions:
            ctxs = retr.retrieve(q["question"], cfg["top_k"])
            ans = generate_answer(q["question"], ctxs)
            samples.append({
                "question_id": q["id"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "contexts": ctxs,
                "answer": ans,
            })
            print(f"  {q['id']} answered ({len(ans)} chars)", flush=True)
        results[vname] = {"config": cfg, "n_chunks": len(chunks), "samples": samples}

    # ---- ragas scoring (collections metrics score samples directly)
    from ragas.metrics.collections.faithfulness import Faithfulness
    from ragas.metrics.collections.context_precision import ContextPrecisionWithReference
    from ragas.metrics.collections.answer_relevancy import AnswerRelevancy

    llm = GeminiInstructorLLM()
    emb = TfidfEmbeddings()
    # pre-fit embeddings on all chunk texts + all questions for a stable vocabulary
    prefit_texts = []
    for vname, cfg in VARIANTS.items():
        prefit_texts += build_chunks(passages, cfg["chunk_size"], cfg["overlap"])
    prefit_texts += [q["question"] for q in questions]
    emb.prefit(prefit_texts)

    faith = Faithfulness(llm=llm)
    ctxp = ContextPrecisionWithReference(llm=llm)
    ansr = AnswerRelevancy(llm=llm, embeddings=emb)

    for vname in VARIANTS:
        print(f"[{datetime.now().isoformat()}] scoring {vname}", flush=True)
        samples = results[vname]["samples"]
        f_in = [{"user_input": s["question"], "response": s["answer"],
                 "retrieved_contexts": s["contexts"]} for s in samples]
        c_in = [{"user_input": s["question"], "reference": s["ground_truth"],
                 "retrieved_contexts": s["contexts"]} for s in samples]
        a_in = [{"user_input": s["question"], "response": s["answer"]} for s in samples]
        f_scores = [r.value for r in faith.batch_score(f_in)]
        c_scores = [r.value for r in ctxp.batch_score(c_in)]
        a_scores = [r.value for r in ansr.batch_score(a_in)]
        per_q, mf, mc, ma = [], 0.0, 0.0, 0.0
        for i, s in enumerate(samples):
            f, c, a = float(f_scores[i]), float(c_scores[i]), float(a_scores[i])
            mf += f; mc += c; ma += a
            per_q.append({
                "question_id": s["question_id"],
                "faithfulness": round(f, 4),
                "context_precision": round(c, 4),
                "answer_relevancy": round(a, 4),
            })
        n = len(samples)
        results[vname]["scores"] = per_q
        results[vname]["means"] = {
            "faithfulness": round(mf / n, 4),
            "context_precision": round(mc / n, 4),
            "answer_relevancy": round(ma / n, 4),
        }
        print(f"  means: {results[vname]['means']}", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "description": (
                "14 passages of public regulatory text: sections 1-10 of the TBS "
                "Directive on Automated Decision-Making (publications.gc.ca, catalogue "
                "BT48-31-2021E-PDF) and a professional summary of OSFI Guideline E-23 "
                "model risk management expectations (BLG, Nov 2025). "
                f"{len(questions)} test questions with hand-written reference answers."
            ),
            "n_passages": len(passages),
            "n_questions": len(questions),
        },
        "retriever": "sklearn TF-IDF (sublinear_tf, 1-2 grams, english stop words), cosine similarity",
        "generator": f"Gemini free tier via gemini skill CLI, model {MODEL}",
        "judge": "Ragas 0.4.3 metrics (faithfulness, context_precision_with_reference, answer_relevancy); "
                 f"LLM judge = {MODEL} via custom instructor-style wrapper; "
                 "embeddings = local TF-IDF (deterministic, CPU-only)",
        "variants": results,
    }
    (WORK / "rag_benchmark.json").write_text(json.dumps(report, indent=2))
    print("saved", WORK / "rag_benchmark.json", flush=True)


if __name__ == "__main__":
    main()
