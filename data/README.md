# Data format

Use a UTF-8 JSON list of records with `text` and `entities`. Character offsets are half-open `[start, end)`. Entity `text`, when provided, must match the indicated substring exactly. Entity types are `taxa`, `location`, `section`, `strata`, `lithology`, `facies`, and `age`.

See `example.json` for a self-created example. Do not use the format example as an experimental dataset.

`corrected_test_gold.json` contains 188 manually annotated texts and 1,281 entities. It replaces the older 1,348-entity annotation file from the previous repository version. These texts were also used for early stopping and model selection in the recorded experiments; use a separate development set and independent final evaluation set for new studies.

The original and augmented training corpora are released at https://huggingface.co/datasets/xhd521/LLMCPNER-data under the fixed tag `revised-2026-11-20`. They are not duplicated in this GitHub repository. The domain dictionary is not bundled. Download the recorded inputs or supply your own files at the paths in the README. New generation is not an exact reconstruction of the historical training corpus. Annotation totals are stored records before tokenizer/span-width filtering.
