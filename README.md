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

I build **LLM systems** — RAG pipelines, synthetic data generators, and tools that grade model output (LLM-as-a-judge). I also fine-tune **audio models** to make music.

- 🧪 **Focus:** better retrieval, clean synthetic data, honest evals, and fine-tuning you can repeat
- 🎧 **Niche:** AI for music production
- 🌐 **Portfolio & blog:** [metzgerdev.github.io/Drum-Machine](https://metzgerdev.github.io/Drum-Machine/#/home)

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

### 🚀 Featured Projects

| Project | What it does |
|---|---|
| **[rag-pipeline-research](https://github.com/metzgerdev/rag-pipeline-research)** | RAG over arXiv papers, tested on `open_ragbench`. It compares three ways to retrieve — keyword (BM25), dense (BGE/E5), and hybrid — at two levels: which paper, and which section. Hybrid finds the right paper every time (recall@10 = 1.0); dense search wins when I need the exact section. Each answer cites its sources with a confidence score, and an LLM judge grades it on relevance, accuracy, completeness, and citations. Runs from the command line (ingest / serve / eval) or a Streamlit page.<br><br>🧰 **Stack:** Python · FAISS · BGE / E5 embeddings · BM25 · OpenAI / OpenRouter · Pydantic · Streamlit |
| **[synthetic-data-pipeline](https://github.com/metzgerdev/synthetic-data-pipeline)** | A six-stage pipeline that writes and checks synthetic DIY-repair Q&A: Generate → Quality Gate → Human Review → LLM Judge → Analysis → Iterate. Llama-3.3-70B writes each item to a Pydantic schema, and semantic dedup drops near-copies. Seven automated checks and a six-point pass/fail rubric (completeness, safety, tools, scope, clarity, usefulness) gate the rest. I tune a Llama-3.1-8B judge against human labels across four prompts until it agrees with people at least 80% of the time. Prompts live in SQLite, so I can edit them while it runs.<br><br>🧰 **Stack:** Python · Groq (Llama 3.3 70B / 3.1 8B) · Pydantic · sentence-transformers · SQLite · Streamlit · Matplotlib |
| **[rag-pipeline](https://github.com/metzgerdev/rag-pipeline)** | A test bench for RAG on long documents like SEC 10-Ks. It runs every mix of 3 ways to split text (sentence, sliding-window, semantic) and 4 ways to retrieve (BM25, dense, hybrid/RRF, LLM-reranked), then reports Recall@K, Precision@K, MRR, MAP, nDCG, and speed. `text-embedding-3-large` with dense search scores best (Recall@5 0.84 at ~410ms). `bge-large` needs no API and answers in ~110ms (Recall@5 0.70–0.76).<br><br>🧰 **Stack:** Python · FAISS · OpenAI `text-embedding-3-large` · BGE-large · BM25 · Jupyter |
| **[llm-resume-coach](https://github.com/metzgerdev/llm-resume-coach)** | Scores how well a resume fits a job. Two scorers run side by side on the same six measures: a fixed-rule labeler you can check by hand (Jaccard skill overlap, years-of-experience gap, seniority rank) and an LLM judge. Together they show exactly where the rules and the model disagree. An 11-point check catches bad data — broken timelines, odd GPAs, inflated "expert" claims. It also builds balanced test data across 5 industries and 5 fit levels, offers four prompts to compare, and logs every prompt or threshold change with before-and-after numbers.<br><br>🧰 **Stack:** Python · Groq (Llama) · Pydantic · REST API · Streamlit |
| **[vocal-emotion-finetune](https://github.com/metzgerdev/vocal-emotion-finetune)** | Adapts `microsoft/wavlm-base-plus` to classify perceived vocal emotion on RAVDESS, as a controlled two-stage experiment on speaker-disjoint splits (no actor leaks across train/val/test). First a frozen-encoder baseline trains only the classifier head; then I unfreeze the top 2 of 12 transformer blocks with discriminative learning rates. Partial fine-tuning lifts test macro-F1 from 0.48 to 0.61 (UAR 0.49 → 0.61) and fixes 42 held-out clips against 7 regressions. I also fit post-hoc temperature scaling on validation and judge it on the held-out test — then reject it, because it worsened calibration. Checkpoints are picked by validation macro-F1, and every run writes its own metrics and report.<br><br>🧰 **Stack:** Python · PyTorch · WavLM · Hugging Face Transformers · scikit-learn · SciPy · uv |
| **[MusicGen-Finetune](https://github.com/metzgerdev/MusicGen-Finetune)** | I fine-tune `facebook/musicgen-small` toward one production sound. The training set is 50–100 ten-second clips, each with a short, steady caption. The repo does the data work — raw audio to normalized clips to JSONL manifests — ready for AudioCraft's `MusicGenSolver`. I judge the result by ear, A/B against the base model.<br><br>🧰 **Stack:** Python · PyTorch · Meta AudioCraft · Hugging Face · MusicGen |
| **[Drum-Machine](https://github.com/metzgerdev/Drum-Machine)** | My portfolio and blog, built as three Web Audio apps in React 19 and TypeScript (Vite, Bun, GraphQL/TanStack Query): a TR-909-style step sequencer, a small DAW to arrange patterns, and a music player. The player's VU meter uses K-weighting and RMS to match how loud a track feels, not just its peaks. The audio engine sits apart from React, so sample timing runs on the Web Audio clock and never waits for a render.<br><br>🧰 **Stack:** React 19 · TypeScript · Vite · Bun · GraphQL · TanStack Query · Web Audio API |

---

<details>
<summary>📊 GitHub Stats</summary>
<br/>
<p align="center">
  <img src="https://github-readme-stats.vercel.app/api?username=metzgerdev&show_icons=true&hide_border=true&theme=default" alt="GitHub stats" height="165"/>
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=metzgerdev&layout=compact&hide_border=true&theme=default" alt="Top languages" height="165"/>
</p>
</details>

<p align="center"><i>Open to applied LLM and audio-AI work.</i></p>
