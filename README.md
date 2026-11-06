# IMF Country Report → AI Pipeline → Financial Stability Dashboard (Guatemala)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Navyanalli/IMF-Guatemala-LLM-Pipeline/blob/main/Guatemala_IMF.ipynb)

MSc Data Science coursework (7PAM2032, University of Hertfordshire) – Assignment 1: LLM coding case study.

A Google Colab pipeline that reads an IMF Article IV staff report (PDF) and uses Large Language Models to produce summaries, structured JSON indicators, thematic counts, grounded RAG answers and a 4-panel financial-stability dashboard.

**Report used:** IMF Country Report No. 26/217 – *Guatemala: 2026 Article IV Consultation – Press Release and Staff Report* (August 2026).

## How to run

1. Click **Open in Colab** above.
2. Click **Runtime → Run all**.

That's it. No files have to be uploaded and Google Drive is not used:

- The report PDF and my saved Gemini answers (`llm_cache.json`) are downloaded automatically from this repository.
- Because every Gemini answer is cached, the notebook reproduces the results in my report **without an API key**.
- *Optional:* to send new prompts, add a free Gemini key (<https://aistudio.google.com/apikey>) in Colab's 🔑 **Secrets** panel as `GEMINI_API_KEY`. The key is never written in the notebook.

It also runs locally in Jupyter (Python 3.10+): `pip install -r requirements.txt`, then open `Guatemala_IMF.ipynb`.

## Pipeline

```
PDF → cleaning → sentence-aware chunks → Gemini Flash (extract / summarise)
    → Gemini Flash-Lite judge + Python checks → JSON files
    → local embeddings + FAISS (RAG) → pd.melt → 4-panel dashboard (300 dpi)
```

| Section | What it does |
|---|---|
| 0. Setup | Configuration, Gemini client with model fallback, one cached helper for every LLM call |
| Task 1 | PDF extraction and cleaning, token- and sentence-aware chunking, section summaries (<100 words), map-reduce executive summary, LLM-as-a-Judge plus Python number checks |
| Task 2 | 7-theme distribution, strengths and vulnerabilities with verbatim evidence, indicator JSON with fallbacks, time series, judge and source audits |
| Task 3 | Dense embeddings (MiniLM vs BGE) with FAISS `IndexFlatL2`, retrieval test set, grounded answers with citation checks |
| Extension | Multi-Query RAG vs single-query RAG |
| Task 4 | `pd.melt` reshaping, 4-panel dashboard, judge-score figure |

## Design choices

- **Two model roles** – the extractor (Gemini Flash) and the judge (Gemini Flash-Lite) are different models, so outputs are never marked by the model that wrote them.
- **Local disk caching** – LLM answers, embeddings and FAISS indexes are saved, so re-runs make no repeat API calls.
- **Model fallback** – if a model is unavailable or rate-limited (404 / 429 / 503), the next one in the list is used and recorded.
- **Fallback values** – indicators not stated in the report are set to a clean default and the reason is logged.
- **Portable** – no Google Drive, no hard-coded local paths, no secret keys in the code.

## Outputs

Created in `imf_outputs/` when the notebook runs:

- `chunks.json`, `sections.json`, `task1_summaries.json`
- `indicators.json`, `findings.json`, `task2_outputs.json`
- `task3_rag.json`, `task4_long_format.json`
- `figures/guatemala_dashboard.png`, `figures/judge_diagnostics.png`

## Tools

Python · pypdf · tiktoken · Google Gemini API (`google-genai`) · sentence-transformers (`all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`) · FAISS · pandas · matplotlib

## Licence

Code released under the [MIT License](LICENSE). The IMF report is © International Monetary Fund and is included only so the notebook can run for assessment.

## References

- International Monetary Fund (2026). *Guatemala: 2026 Article IV Consultation – Press Release and Staff Report.* IMF Country Report No. 26/217. Washington, DC: IMF.
- Google (2026). Gemini API documentation. https://ai.google.dev
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP*.
- Xiao, S. et al. (2023). C-Pack: Packaged resources to advance general Chinese embedding (BGE). *arXiv:2309.07597*.
- Johnson, J., Douze, M. & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*.
- Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS*.
