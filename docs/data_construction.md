# Constructing supervision data

These commands expose the recovered clustering, annotation, segmentation, and three augmentation strategies with portable paths and environment-based API configuration. They do not redistribute source PDFs. MinerU PDF parsing and manual corpus screening occur before these commands.

## 1. Cluster and sample extracted abstracts

The input is an object containing an `abstracts` list. Each record has `filename`, `category`, and `abstract`; see `data/abstracts_example.json`. Use a corpus with at least 100 records for the default 100 clusters. Ensure output directories exist.

```bash
mkdir -p outputs
python scripts/cluster_sample.py \
  --abstracts-file data/abstracts.json \
  --output-file outputs/cluster_samples.json \
  --k 100 --samples 30 --model all-MiniLM-L6-v2
python scripts/flatten_clusters.py \
  --input outputs/cluster_samples.json \
  --output data/cluster_abstracts.json
```

K-means uses `random_state=42` and `n_init=10`. Sampling draws at most 30 entries from each cluster. The release exposes a default sampling seed of 42; the recovered historical script did not explicitly seed Python's sampling generator. The saved historical sample, not a newly drawn sample, determines exact corpus membership.

## 2. Annotate and segment

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
export OPENAI_BASE_URL='https://api.openai.com/v1'
export LLM_MODEL='gpt-5.4-mini'

python scripts/annotate.py \
  --input data/cluster_abstracts.json \
  --fewshot data/fewshot.json \
  --output-dir outputs/annotation \
  --model "$LLM_MODEL"

PALEO_INPUT_PATH=outputs/annotation/training_data.json \
PALEO_OUTPUT_PATH=data/training_original.json \
python scripts/segment.py
```

`data/fewshot.json` must use the same JSON annotation schema as `data/example.json`. The annotation script retains recovered prompts, optional verification, confidence filtering, and checkpointed API requests. These original annotation instructions are weak-supervision prompts, not the manually revised evaluation guideline. Recovered training annotations are not retrospectively rewritten to match the evaluation reference.

Segmentation uses spaCy `en_core_web_sm` when available and a rule-based fallback otherwise; the choice can affect resulting instances. Check generated offsets and preserve original source identifiers. API responses and manually curated resources influence the generated corpus, so the historical counts in the README are not guaranteed outcomes of new API calls.

## 3. Generate three augmentation pools

Provide a JSON entity pool mapping internal labels to lists of entity names, following `data/entities_example.json`. Use the curated domain dictionary for a paper-comparable run; the demonstration pool is only for checking file format.

```bash
python scripts/augment.py \
  --input data/training_original.json \
  --entity-pool data/domain_entity_pool.json \
  --replacement 3334 --rewriting 3333 --combination 3333 \
  --model "$LLM_MODEL" \
  --output data/training_augmented.json
```

The three strategy functions and their prompts are retained from the recovered augmentation source. The release wrapper requests exact strategy counts, checks offsets, saves accepted examples after each success, and fails if a generation budget cannot be met. Partial output is retained for inspection; automatic API-generation resumption is not implemented by this wrapper. It uses sequential requests for a clear, reproducible control flow; no requests run until invoked by the user.

Entity replacement uses dictionary substitutions; rewriting and combination call the selected LLM. To use an open-weight model, set `OPENAI_BASE_URL` to its OpenAI-compatible endpoint and pass its served model name. This service switch alone does not reproduce the paper's archived Qwen experiment.

## 4. Prepare training inputs

Review generated annotations and run `scripts/merge_training.py` as shown in the main README. It validates stored offsets and labels before merging. The exact final training view in the paper additionally followed the recorded corpus filtering; preserve the archived final input files when reproducing its numbers.
