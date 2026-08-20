"""Project-wide defaults. Values can be overridden with environment variables."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("PALEO_NER_DATA_DIR", ROOT_DIR / "data"))
OUTPUT_DIR = Path(os.getenv("PALEO_NER_OUTPUT_DIR", ROOT_DIR / "outputs"))
MODEL_PATH = os.getenv("PALEO_NER_MODEL", "allenai/scibert_scivocab_uncased")

ENTITY_TYPES = ["taxa", "location", "section", "strata", "lithology", "facies", "age"]
LABEL2ID = {"O": 0, **{label: i + 1 for i, label in enumerate(ENTITY_TYPES)}}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)
TYPE_MAPPING = {
    "litology": "lithology",
    "taxon": "taxa",
    "geological_age": "age",
}

MAX_LENGTH = 512
MAX_SPAN_LENGTH = 12
NEG_POS_RATIO = 5
BATCH_SIZE = 8
DEVICE = "cuda"

