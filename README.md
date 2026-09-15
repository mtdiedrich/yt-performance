# yt-performance

**Predicting YouTube video performance from titles, thumbnails, and transcripts — with the numbers up front.**

## Results (the short version)

On a 1.15M-video corpus, predicting **channel-relative** view counts (per-channel MinMax-normalized):

| Approach | Test R² |
|---|---|
| CatBoost + roberta-large title embeddings | 0.062 |
| CatBoost + fine-tuned MiniLM title embeddings | **0.221** |
| NN on transcript topic clusters | ~0.00 (negative result, documented) |
| AutoGluon multimodal (title + thumbnail) | run, metrics not persisted |

Full table, setup, and findings: [results/MODEL_COMPARISON.md](results/MODEL_COMPARISON.md)

**Takeaway:** fine-tuning the title embedding model was the single largest win (R² 0.06 → 0.22); multimodal and transcript-based features never beat titles alone in persisted runs.

## What this repo is

A consolidation of three previously scattered pieces of one project:

- `src/datahub/` — thumbnail-analytics DB layer (SQLite + AWS Rekognition labels/colors/quality), from the `banshee` repo
- `src/thumbnail/` — Rekognition integration, from `YTModel`
- `notebooks/title-embeddings/` — the winning approach: CatBoost over fine-tuned sentence-transformer title embeddings, incl. per-channel normalization and a held-out benchmark on the author's own channel
- `notebooks/multimodal/` — AutoGluon multimodal (title + ResNet50 thumbnail), thumbnail download pipelines, TF-IDF/spaCy/XGBoost baselines
- `notebooks/high-performance/` — transcript parsing, topic clustering, and the "high performance intro" NN experiments (negative results preserved)
- `notebooks/thumbnail-og/` — earlier-generation thumbnail/title experiments (historical)
- `notebooks/rag/` — RAG exploration over channel documents (mpnet + Qwen)

## Pipeline

```
YouTube API / scraping → SQLite/DuckDB → Rekognition (thumbnails) → sentence embeddings (titles/descriptions)
                                    ↘ transcripts → topic clusters
→ CatBoost / AutoGluon / NN models → channel-relative view prediction
```

## Status & next steps

- [ ] Persist and publish the missing scores (OpenAI-embedding CatBoost, AutoGluon leaderboard)
- [ ] Reproduce the R² 0.22 headline in a clean script (not only a Colab notebook)
- [ ] Backtest the "predicted vs. actual" table on the author's own channel as a demo artifact

## Provenance

Merged 2026-09 from: `banshee` (GitHub), `ai-ml/YTModel` (local), and a set of Colab notebooks. Originals left untouched; this repo is the canonical home going forward. Raw data (videos, thumbnails, DBs, parquet) is not committed — see `.gitignore`.
