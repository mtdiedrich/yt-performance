# Model Comparison — YouTube View Prediction

All numbers verbatim from persisted notebook outputs (`notebooks/title-embeddings/NeedForSpeed.ipynb` experiment log, `notebooks/high-performance/*` training logs). Nothing here was re-run for this table.

## Setup

- **Corpus:** 1,150,532 YouTube videos (gaming niche, `need_for_speed_30D` / `nfs_m` SQLite DBs)
- **Split:** 92,040 train / 11,505 val / 11,506 test, plus 15 videos from the author's own channel held out as a personal benchmark
- **Target:** view counts **MinMax-normalized within each channel** — the task is "predict a video's performance relative to its own channel," not raw views

## Results

| Model / features | Test R² | Source |
|---|---|---|
| CatBoost + `all-roberta-large-v1` title embeddings (16k iters, depth 8) | 0.062 | NeedForSpeed.ipynb, 2024-08-21 run log |
| CatBoost + `all-roberta-large-v1` (4 more configs) | 0.054–0.062 | same |
| CatBoost + fine-tuned `all-MiniLM-L6-v2` (custom embeddings) | 0.217 | NeedForSpeed.ipynb, final run |
| CatBoost + MiniLM, tuned (1024 iters, depth 8, lr 0.05, l2 100, rs 10) | **0.221** | NeedForSpeed.ipynb, grid-sampled config |
| PyTorch NN + transcript topic-cluster features (YTModel) | val R² ≈ 0.00–0.02, degrades negative with training | `0 High Performance Model 3.ipynb`, `4 Model With Cluster Analysis.ipynb` |
| AutoGluon multimodal (title + thumbnail) | *not persisted* — widget-driven runs, leaderboard not saved | `autogluon_model.ipynb` |
| CatBoost + OpenAI 3072-d embeddings (native embedding features) | *not persisted* — final `model.score` outputs not saved | `nfs_m.ipynb` |

## Findings

1. **Title semantics carry the signal.** Fine-tuning the embedding model (roberta-large → custom MiniLM) was the single biggest improvement: test R² 0.06 → 0.22.
2. **Multimodal and transcript features did not beat titles alone** in persisted runs. The transcript-cluster NN never exceeded val R² 0.02 and overfit into negative territory — a negative result worth keeping.
3. **Channel-relative prediction is genuinely hard.** R² 0.22 on this normalized target is the honest headline; raw-view prediction would inflate the number while meaning less.
4. Reproducibility gap to fix: the two most recent model families (OpenAI-embedding CatBoost, AutoGluon multimodal) logged no final metrics. Any rerun must persist scores to `results/`.
