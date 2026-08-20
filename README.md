# LLMCPNER

Official code for the manuscript **"LLMCPNER: Integrating Large Language
Models and Curriculum Learning for Paleontological Named Entity Recognition"**,
currently under review.

LLMCPNER recognizes seven entity types in paleontological literature: taxa,
locations, sections, strata, lithology, facies, and geological age. The method
combines a SciBERT span classifier, multi-model voting, and confidence-weighted
curriculum learning.

## Repository structure

```text
LLMCPNER/
|-- src/
|   |-- config.py       # labels, paths, and default settings
|   |-- dataset.py      # span construction and batching
|   |-- models.py       # SciBERT span classifier
|   `-- train.py        # complete training and evaluation pipeline
|-- data/
|   |-- corrected_test_gold.json
|   `-- README.md
|-- requirements.txt
`-- LICENSE
```

This repository does not include baseline implementations, ablation code,
training data, model weights, checkpoints, or experiment outputs.

## Installation

Python 3.10 or later is recommended. A CUDA-capable GPU is recommended for
training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default encoder is `allenai/scibert_scivocab_uncased`. A local model path
or another compatible Hugging Face identifier can be supplied through:

```bash
export PALEO_NER_MODEL=/path/to/scibert
```

## Data

The manually corrected test set is included at
`data/corrected_test_gold.json`. It contains 188 texts and 1,348 entities and
is used only for final evaluation. It must not be used for training,
curriculum construction, threshold selection, or model selection.

Training data are not distributed in this repository. The expected format is
a UTF-8 JSON list with half-open character offsets `[start, end)`:

```json
[
  {
    "text": "Trilobites occur in the Wheeler Shale.",
    "entities": [
      {"start": 0, "end": 10, "entity_type": "taxa"},
      {"start": 24, "end": 37, "entity_type": "strata"}
    ]
  }
]
```

Valid entity types are `taxa`, `location`, `section`, `strata`, `lithology`,
`facies`, and `age`.

## Running the complete experiment

The weakly supervised training set is **not included in this repository**.
To retrain the model, provide the path to a separately obtained training file:

```bash
python src/train.py \
  --train-file /path/to/private_training_data.json \
  --test-file data/corrected_test_gold.json \
  --gpu 0
```

Outputs are written to `outputs/complete_experiment/` by default and include
the final metrics, predictions, logs, confidence scores, and trained weights.
Use `python src/train.py --help` to view configurable settings.

## Model checkpoint

The final model checkpoint is available on Hugging Face:

https://huggingface.co/xhd521/LLMCPNER

It can be downloaded with:

```bash
hf download xhd521/LLMCPNER model.pt --local-dir checkpoints/LLMCPNER
```

The checkpoint uses the custom span-classification implementation in this
repository and is not a drop-in `AutoModel.from_pretrained()` model.

## Code availability

The code supporting this manuscript is available at:

https://github.com/goodXHD/LLMCPNER-Integrating-Large-Language-Models-and-Curriculum-Learning-for-Paleontological-Named-Entity

## License

The original code is released under the MIT License. This license does not
automatically apply to external training data or pretrained models.
