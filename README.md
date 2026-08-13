<h1 align="center">Hi, I'm Nam Dao 👋</h1>

<p align="center">
  <b>AI & Full-Stack Engineer</b> · Los Angeles<br/>
  Previously @ NGINX · Rescale · Nintendo
</p>

<p align="center">
  <a href="https://metzgerdev.github.io/Drum-Machine/#/home"><img src="https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Portfolio"/></a>
  <a href="https://www.linkedin.com/in/nam-dao"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
</p>

---

I build **LLM and audio systems**, and I evaluate them properly: held-out splits that
don't leak, a baseline I have to beat before claiming anything, and the results that went
the wrong way reported rather than buried.

- 🧪 **Focus:** retrieval, synthetic data, honest evals, and fine-tuning you can repeat
- 🎧 **Niche:** AI for music production
- 🌐 **Portfolio & blog:** [metzgerdev.github.io/Drum-Machine](https://metzgerdev.github.io/Drum-Machine/#/home)

### How I evaluate

- **Split so it can't leak.** Speaker-disjoint splits for audio, held-out conditioning for
  generative models — not a random shuffle over correlated samples.
- **Beat a baseline, not a vacuum.** Every number below is next to the thing it beat, and
  the cost it paid to win.
- **Report what failed.** Post-hoc calibration that made calibration worse, a GPU slower
  than the CPU, a preference method whose effect I couldn't distinguish from noise.

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

### 🚀 Featured

#### [midi_gpt](https://github.com/metzgerdev/midi_gpt) — a 0.687M-parameter language model that composes UK garage MIDI

**Conditioning does the work, so the model can be tiny.** It emits one token per sixteenth
note — a pitch, a sustain, or a rest, 65 symbols in all — and at every step receives two
things it never has to infer: a kick-pocket grid lifted from a drum loop, and the current
chord as a 12-dimensional pitch-class mask. I train it from scratch on a mined MIDI corpus
where every phrase appears in all twelve keys with notes and chord rotated together, which
makes absolute pitch carry zero information: reading the chord track is the only way to
lower the loss. Then I tune it to my taste — I edit the output in Ableton and DPO learns
the gap between what it wrote and what I kept.

*What didn't work:* MPS ran **10× slower than CPU** (632 ms vs 60 ms per candidate) — the
tensors are too small to amortize launch overhead. And DPO's effect on generation sat
**inside seed noise** at 16 preference pairs, measured against a noise floor built by
half-splitting the model's own samples; the in-sample reward margin looked excellent and
meant nothing.

Eight bars in ~4.5 s on a laptop CPU. [Pipeline diagram](https://github.com/metzgerdev/midi_gpt/blob/main/pipeline.html) · `uv sync --frozen && uv run python make_track.py`
<sub>Python · PyTorch · mido · librosa · uv</sub>

#### [vocal-emotion-finetune](https://github.com/metzgerdev/vocal-emotion-finetune) — controlled two-stage fine-tune on RAVDESS

**Partial fine-tuning lifts test macro-F1 from 0.48 to 0.61 (UAR 0.49 → 0.61)** on
speaker-disjoint splits, with no actor appearing in more than one split. Stage one trains
only the classifier head on a frozen `microsoft/wavlm-base-plus`; stage two unfreezes the
top 2 of 12 transformer blocks with discriminative learning rates. The change fixes 42
held-out clips against 7 regressions, so the gain isn't a wash of trades.

*What didn't work:* I fit post-hoc temperature scaling on validation, evaluated it on the
held-out test — and **rejected it, because it made calibration worse**. Checkpoints are
selected on validation macro-F1 only, never on test.

[Fine-tune writeup](https://github.com/metzgerdev/vocal-emotion-finetune/blob/main/docs/partial_finetune.md) · [calibration result](https://github.com/metzgerdev/vocal-emotion-finetune/blob/main/docs/calibration.md) · [baseline](https://github.com/metzgerdev/vocal-emotion-finetune/blob/main/docs/baseline.md)
<sub>Python · PyTorch · WavLM · Hugging Face · scikit-learn · SciPy · uv</sub>

#### [rag-pipeline](https://github.com/metzgerdev/rag-pipeline) — full-factorial retrieval bench on long documents

**Dense retrieval reaches recall@5 0.84, against 0.70 hybrid and 0.58 BM25 — and pays 50×
the latency for it** (413 ms vs 8 ms). I run all 27 combinations of 3 chunking strategies
(sentence, sliding-window, semantic) × 3 embeddings (`text-embedding-3-large`,
`bge-large`, `bge-small`) × 3 retrieval methods (BM25, dense, hybrid/RRF) over SEC 10-Ks,
scoring Recall@K, Precision@K, MRR, MAP, nDCG and latency. The interesting result is the
trade, not the winner: `bge-large` needs no API and answers in **110 ms at recall 0.76**,
which is the configuration I'd actually ship.

[Full results grid](https://github.com/metzgerdev/rag-pipeline/blob/main/grid_results.csv) (27 configurations) · [notebook](https://github.com/metzgerdev/rag-pipeline/blob/main/pipeline.ipynb)
<sub>Python · FAISS · OpenAI text-embedding-3-large · BGE-large · BM25 · Jupyter</sub>

---

### Also

- **[rag-pipeline-research](https://github.com/metzgerdev/rag-pipeline-research)** — RAG over arXiv papers on `open_ragbench`, comparing keyword, dense and hybrid retrieval at both paper and section level. Hybrid finds the right paper every time (recall@10 = 1.0); dense wins on locating the exact section. Answers cite sources with a confidence score, graded by an LLM judge.
- **[synthetic-data-pipeline](https://github.com/metzgerdev/synthetic-data-pipeline)** — six stages from generation to iteration for synthetic DIY-repair Q&A. Seven automated checks plus a six-point rubric gate the output, and I tune a Llama-3.1-8B judge against human labels across four prompts until it agrees with people ≥80% of the time.
- **[llm-resume-coach](https://github.com/metzgerdev/llm-resume-coach)** — scores resume/job fit with two scorers side by side: a hand-checkable rule labeler (Jaccard skill overlap, experience gap, seniority rank) and an LLM judge, so you can see exactly where they disagree. Logs every prompt and threshold change with before-and-after numbers.
- **[MusicGen-Finetune](https://github.com/metzgerdev/MusicGen-Finetune)** — fine-tunes `facebook/musicgen-small` toward one production sound on 50–100 captioned clips, with the raw-audio-to-JSONL data path ready for AudioCraft's `MusicGenSolver`. Judged by ear, A/B against base.
- **[Drum-Machine](https://github.com/metzgerdev/Drum-Machine)** — my portfolio and blog: three Web Audio apps in React 19 and TypeScript. The audio engine sits outside React so sample timing runs on the Web Audio clock and never waits for a render.

---

<p align="center"><i>Looking for applied LLM or audio-ML work — model evaluation, fine-tuning pipelines, and generative audio tooling.</i></p>
