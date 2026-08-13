<h1 align="center">Hi, I'm Nam Dao 👋</h1>

<p align="center">
  <b>AI Engineer</b> · Los Angeles<br/>
  Previously @ NGINX · Rescale · Nintendo
</p>

<p align="center">
  <a href="https://metzgerdev.github.io/Drum-Machine/#/home"><img src="https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Portfolio"/></a>
  <a href="https://www.linkedin.com/in/nam-dao"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
</p>

---

I build **LLM systems** with a niche in audio applications.

---

### 🛠️ Tech

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-005571?style=flat-square&logo=meta&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)

---

#### [midi_gpt](https://github.com/metzgerdev/midi_gpt) — a small language model that generates MIDI

I built a small language model, GPT-2 transformer architecture, that generates MIDI from a
drum loop and a key as the prompt. I trained it on a corpus of MIDI patterns from house and
UK garage. The corpus was expanded deterministically to generate additional synthetic data
for training. I then performed SFT and DPO to fine-tune the model to my personal taste,
and a DPO script is provided so the end user can perform additional fine-tuning.

Because MIDI is a compact symbolic representation of music, the model is small enough to
run inference on CPU in a couple of seconds. It has 0.687M parameters and writes eight bars
in about 4.5 seconds.

**Tech Stack:** Python · PyTorch · mido · librosa · NumPy · SciPy · uv

#### [vocal-emotion-finetune](https://github.com/metzgerdev/vocal-emotion-finetune) — two-stage fine-tune for vocal emotion

I built a vocal emotion classifier by fine-tuning `microsoft/wavlm-base-plus` in two stages. The first
stage trains only the classifier head on a frozen encoder. The second unfreezes the top 2
of 12 transformer blocks with discriminative learning rates. Test macro-F1 goes from 0.48
to 0.61 and UAR from 0.49 to 0.61, fixing 42 held-out clips against 7 regressions, so the
gain is not a wash of trades.

**Tech Stack:** Python · PyTorch · WavLM · Hugging Face Transformers · scikit-learn · SciPy · uv

#### [rag-pipeline](https://github.com/metzgerdev/rag-pipeline) — a retrieval test bench for long documents

I built a RAG pipeline for retrieval of long documents like SEC 10-Ks. I ran 27
experiments across combinations of chunking strategies (sentence, sliding-window, semantic), embeddings
(`text-embedding-3-large`, `bge-large`, `bge-small`) and retrieval methods (BM25, dense,
hybrid/RRF), scoring Recall@K, Precision@K, MRR, MAP, nDCG and latency.

Dense retrieval reaches recall@5 of 0.84, against 0.70 for hybrid and 0.58 for BM25, and
pays 50× the latency for it, 413 ms against 8 ms. The configuration I would actually ship
is `bge-large`: no API, 110 ms, recall 0.76.

**Tech Stack:** Python · FAISS · OpenAI `text-embedding-3-large` · BGE-large · BM25 · Jupyter

#### [rag-pipeline-research](https://github.com/metzgerdev/rag-pipeline-research) — RAG over arXiv papers

I built a RAG assistant over arXiv papers and evaluated retrieval on `open_ragbench`. It
runs 12 configurations, 3 chunking strategies × 2 embeddings × 2 retrievers, scored at two
levels: which paper, and which section within that paper. Answers cite their sources with
a confidence score, and an LLM judge grades each one from 1 to 5 on relevance, accuracy,
completeness and citation quality.

The paper-level task turned out to be saturated. All 12 configurations reach recall@5 of
1.0 and hybrid ties dense exactly, so the grid cannot separate them at that level and the
differences only appear once I ask which section. The finding worth keeping is about cost:
`bge-small` matches `bge-large` at 18 ms per query against 40 ms, so the smaller model is
the one to run.

**Tech Stack:** Python · FAISS · BGE / E5 embeddings · BM25 · OpenAI / OpenRouter · Pydantic · Streamlit

#### [synthetic-data-pipeline](https://github.com/metzgerdev/synthetic-data-pipeline) — synthetic Q&A with a calibrated judge

I built a six-stage pipeline that writes and checks synthetic DIY-repair Q&A: generate,
quality gate, human review, LLM judge, analysis, iterate. Llama 3.3 70B writes each item
against a Pydantic schema and semantic dedup drops the near-copies. Seven automated checks
and six binary dimensions — completeness, safety, tools, scope, clarity, usefulness — gate
what survives.

The part that matters is the judge. I label a sample by hand first, then tune a
Llama 3.1 8B judge across four prompt variants until it agrees with my labels on at least
80% of every dimension, not just on average, because a judge that is right overall can
still be wrong on the one dimension I care about. Prompts live in SQLite so I can edit
them while the pipeline runs.

**Tech Stack:** Python · Groq (Llama 3.3 70B / 3.1 8B) · Pydantic · sentence-transformers · SQLite · Streamlit · Matplotlib

#### [llm-resume-coach](https://github.com/metzgerdev/llm-resume-coach) — two scorers on the same rubric

I built a scorer for how well a resume fits a job, with two scorers running side by side
on the same six binary dimensions: a deterministic labeler you can check by hand, using
Jaccard skill overlap, years-of-experience gap and seniority rank, and an LLM judge.
Running both means every disagreement is visible and attributable to one of them, instead
of trusting a single number.

Eleven per-item checks catch bad data before any scoring — broken timelines, implausible
GPAs, inflated expertise claims. The test set is generated evenly across 5 industries × 5
fit levels so the evaluation is not weighted toward the easy cases, and every prompt or
threshold change is logged with before-and-after numbers.

**Tech Stack:** Python · Groq (Llama) · Pydantic · REST API · Streamlit
