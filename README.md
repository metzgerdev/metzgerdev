<h1 align="center">Hi, I'm Nam Dao 👋</h1>

<p align="center">
  <b>AI Engineer</b> · Los Angeles<br/>
  Previously @ NGINX · Rescale · Nintendo
</p>

<p align="center">
  <a href="https://metzgerdev.github.io/nam-dao/"><img src="https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Portfolio"/></a>
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

#### [Midi GPT](https://github.com/metzgerdev/midi_gpt) — a small language model that generates MIDI for electronic music

I built a small language model using a GPT-2 transformer architecture that generates MIDI from a
drum loop and key as the prompt. The model is trained on a corpus of MIDI patterns from house and
UK garage. The corpus was expanded deterministically to generate additional synthetic data
for training. I then performed SFT and DPO to fine-tune the model to my personal taste.  A DPO script is provided so the end user can perform additional fine-tuning.

Because MIDI is a compact symbolic representation of music, the model is small enough to
run inference on CPU in a couple of seconds. It has 0.687M parameters and writes eight bars
in a few seconds.  A 1/16 note grid and drum conditioning results in a tight rythymic lock.

**Tech Stack:** Python · PyTorch · mido · librosa · NumPy · SciPy · uv

#### [RAG Pipeline](https://github.com/metzgerdev/rag-pipeline) — eval backed rag system

I built a RAG pipeline for retrieval of long documents like SEC 10-Ks. I ran 27
experiments across combinations of chunking strategies (sentence, sliding-window, semantic), embeddings
(`text-embedding-3-large`, `bge-large`, `bge-small`) and retrieval methods (BM25, dense,
hybrid/RRF).  Eval metrics include Recall@K, Precision@K, MRR, MAP, nDCG and latency.

I generated synthetic QA datasets per chunking configuration with an automated evaluation framework to determine the optimum configuration.

**Tech Stack:** Python · FAISS · OpenAI `text-embedding-3-large` · BGE-large · BM25 · Jupyter

#### [RAG Research Assistant](https://github.com/metzgerdev/rag-pipeline-research) — rag over arXiv papers

I built a RAG-based research assistant ingesting arXiv papers using the `open_ragbench` dataset. Multiple experiments are run with configurations based on 3 chunking strategies × 2 embeddings × 2 retrievers. I use an LLM-as-Judge to evaluate the generated answers based on a four-dimensional rubric. Additionally, a Streamlit web UI allows execution of the entire pipeline.

**Tech Stack:** Python · FAISS · BGE / E5 embeddings · BM25 · OpenAI / OpenRouter · Pydantic · Streamlit

#### [Synthetic Data Pipeline](https://github.com/metzgerdev/synthetic-data-pipeline) — synthetic data generation pipeline with LLM-as-Judge

I built a pipeline for synthetic data generation for a Q&A repair chatbot. The pipeline generates structured repair guidance evaluated for data quality, along with human-in-the-loop labeling. The evaluation metric is LLM-as-Judge agreement with human labeling over six quality dimensions. The generation prompt and judge prompt are iterated until the agreement threshold is met. Visualizations, metrics and log reports are generated to guide prompt adjustment based on empirical observations.

**Tech Stack:** Python · Groq (Llama 3.3 70B / 3.1 8B) · Pydantic · sentence-transformers · SQLite · Streamlit · Matplotlib

#### [LLM Resume Coach](https://github.com/metzgerdev/llm-resume-coach) — resume screening system

I built a resume coach using synthetic data, a rule-based analyzer, and LLM-as-a-Judge evaluation. The system identifies the fit between a resume and job posting, detects quality issues, and provides actionable feedback.

The evaluation pipeline includes data validation and correlation matrices with heatmaps. The system also exposes an API for running the pipeline.

**Tech Stack:** Python · Groq (Llama) · Pydantic · REST API · Streamlit
