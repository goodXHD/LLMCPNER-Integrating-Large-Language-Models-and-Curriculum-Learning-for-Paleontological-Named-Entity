# LLMCPNER

Code for **LLMCPNER: Integrating Large Language Models and Curriculum Learning for Paleontological Named Entity Recognition**.

LLMCPNER constructs weak supervision with a large language model, expands domain entity contexts through three augmentation strategies, and trains a SciBERT span recognizer using multi-voter confidence and cumulative curriculum learning. At inference time, only the trained span recognizer is required.

[中文说明](README_zh.md)

## Method

1. **Representative-text annotation.** Encode candidate abstracts with `all-MiniLM-L6-v2`, apply K-means with 100 clusters (`random_state=42`, `n_init=10`), and sample up to 30 records per cluster for LLM annotation.
2. **Domain data augmentation.** Apply entity replacement, text rewriting, and entity combination. The recorded training corpus contains 3,334 replacement, 3,333 rewriting, and 3,333 combination instances.
3. **Multi-voter confidence.** Train three independently initialized SciBERT span classifiers for two epochs each on 60% bootstrap samples of the initial weakly annotated data. Sampling combines uniform sentence draws and type-balanced draws. Score the supplied positive labels with the three trained voters in evaluation mode, with dropout disabled.
4. **Confidence-guided curriculum.** Rank positive spans within each entity type, using the 70th and 30th confidence percentiles to form high-, medium-, and low-confidence groups. Training progressively includes these groups. An admitted span keeps its confidence weight; a positive span not yet admitted is masked rather than treated as a negative. Sampled negatives always have weight 1.

For a positive span $s$ with automatic label $\tilde c_s$, confidence is

$$
r(s)=\frac{1}{2}\left[\frac{1}{M}\sum_{j=1}^{M}P_j(\tilde c_s\mid s,x)+\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}(\hat c_j(s)=\tilde c_s)\right],\quad M=3.
$$

The weighted cross-entropy is divided by the number of active spans, not the sum of weights. Confidence supports annotation-error screening, but voters trained from the same weak-label source can share systematic errors.

## Entity types and data

| Paper entity type | Internal label |
|---|---|
| Taxa | `taxa` |
| Location | `location` |
| Section | `section` |
| Stratigraphic unit | `strata` |
| Lithology | `lithology` |
| Facies | `facies` |
| Age | `age` |

The internal label `strata` is retained for compatibility with saved data and checkpoints; it denotes the paper's **Stratigraphic unit** category.

| Dataset | Text instances | Entity annotations |
|---|---:|---:|
| Initial LLM-annotated training data | 11,052 | 49,003 |
| Domain-augmented training data | 10,000 | 64,791 |
| Manually annotated evaluation data | 188 | 1,281 |

`data/corrected_test_gold.json` contains the 188 manually annotated texts. Initial and augmented training corpora, source PDFs, domain dictionaries, and model weights are not bundled. `data/example.json` is a short, self-created format example.

The 188 texts were also used for early stopping and model selection in the recorded experiments. Their reported performance is therefore not an independent held-out estimate. For new experiments, supply separate development and final evaluation files.

## Repository structure

```text
configs/full_method.json       # shared defaults
scripts/cluster_sample.py      # semantic clustering and sampling
scripts/flatten_clusters.py    # annotation input preparation
scripts/annotate.py            # LLM annotation and verification prompts
scripts/segment.py             # sentence segmentation and offset mapping
scripts/augment.py             # fixed-count generation for three strategies
scripts/augmentation_strategies.py
scripts/merge_training.py
src/config.py                  # label IDs and span limits
src/dataset.py                 # positive spans and sampled negatives
src/models.py                  # SciBERT span model and decoder
src/voters.py                  # voter training and positive-label scoring
src/aggregate.py               # confidence and curriculum assignment
src/weighting.py               # admission masks and weighted loss
src/train.py                   # staged training and checkpoint selection
src/evaluate.py                # evaluation or unlabeled extraction
src/metrics.py                 # strict and partial matching
src/engine.py, src/runtime.py  # shared helpers
checks/smoke.py                # small, offline functional checks
```

## Installation

Use Python 3.10 or later. The released training commands follow the recorded CUDA/BF16 protocol and require a GPU supporting BF16. CPU evaluation is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Optional: annotation, augmentation, clustering and segmentation dependencies
python -m pip install -r requirements-data.txt
python -m spacy download en_core_web_sm
```

Download SciBERT into a local directory before training. For reproducible runs, record the Hugging Face revision used.

```bash
hf download allenai/scibert_scivocab_uncased --local-dir encoders/scibert
```

API-backed scripts read `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LLM_MODEL` from the environment. Use an OpenAI-compatible service, including a locally served open-weight model if desired. No API calls are made by voter training, main-model training, or inference.

## Train the full method

Provide `data/training_original.json`, `data/training_augmented.json`, and a development file in the format described in [data/README.md](data/README.md). The combined data must stay in the same order during scoring and training.

```bash
python scripts/merge_training.py \
  --original data/training_original.json \
  --augmented data/training_augmented.json \
  --output data/training_combined.json

for member in 0 1 2; do
  python src/voters.py \
    --member "$member" \
    --original-file data/training_original.json \
    --train-file data/training_combined.json \
    --encoder encoders/scibert \
    --output-dir "outputs/voter_${member}" \
    --device cuda:0
done

python src/aggregate.py \
  --voter-dirs outputs/voter_0 outputs/voter_1 outputs/voter_2 \
  --output-dir outputs/confidence

python src/train.py \
  --train-file data/training_combined.json \
  --dev-file data/dev.json \
  --memberships outputs/confidence/memberships.json \
  --encoder encoders/scibert \
  --output-dir outputs/full_method \
  --device cuda:0
```

The example runs voters sequentially on one GPU. They may instead be launched on separate GPUs with distinct output directories.

The final protocol uses seed 13, batch size 8, learning rate $2\times10^{-5}$, AdamW weight decay 0.01, a maximum input length of 512 tokens, and a maximum span width of 12 tokens. Negative sampling uses up to 15 negatives per positive, with a minimum target of 50 negatives when available.

Each of the three curriculum stages trains for at least two epochs and stops after three consecutive epochs without an increase in development strict F1. There is no fixed maximum number of epochs. A stage transition restores the previous stage's best model and matching Adam moments, then restarts the stage-local learning-rate schedule. Each stage warms up for 10% of one epoch and decays to a 0.1 learning-rate floor over a ten-epoch reference horizon. The final checkpoint is the **best checkpoint in stage 3**.

`epoch_metrics.json` records every epoch, and `status.json` records current progress. `stage_summary.json` records actual stage lengths. `resume.pt` stores the last complete epoch, but automatic process resumption is not implemented. Output directories must be new to avoid overwriting existing runs.

Optional controls are available via `--condition ordinary`, `static_confidence`, or `random_cumulative`; the default is `class_cumulative`.

## Evaluation and prediction

```bash
python src/evaluate.py \
  --checkpoint outputs/full_method/model.pt \
  --encoder encoders/scibert \
  --input data/corrected_test_gold.json \
  --output-dir outputs/evaluation \
  --device cuda:0
```

Strict matching requires identical character boundaries and entity types. Partial matching requires the same entity type and character-span intersection-over-union of at least 0.5. Both use one-to-one matching and micro-averaged precision, recall, and F1.

For extraction from unlabeled JSON records containing `text`, add `--unlabeled`. The output `predictions.json` contains half-open character offsets, entity types, entity text, and model scores. SciBERT's 512-token input limit still applies; split longer documents into suitable text instances before inference.

## Reported results

| Method | Strict precision | Strict recall | Strict F1 | Partial F1 |
|---|---:|---:|---:|---:|
| LLMCPNER | 83.08 | 86.26 | 84.64 | 89.54 |

Scores are percentages for the recorded model on the 188-text reference. Re-running API-based data construction can change generated examples and is not guaranteed to reproduce the historical corpus or these exact scores.

See [docs/data_construction.md](docs/data_construction.md) for data-building commands and [docs/reproduction.md](docs/reproduction.md) for the correspondence between the release and archived experiments. Baseline implementations and unrelated exploratory experiments are not included.

## License

The code is released under the MIT License. This license does not automatically cover source publications, external datasets, or pretrained model weights.
