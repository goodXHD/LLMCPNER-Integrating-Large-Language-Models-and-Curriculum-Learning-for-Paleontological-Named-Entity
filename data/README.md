# Data directory

`corrected_test_gold.json` is the manually corrected evaluation set used by the
paper. It must be used for final evaluation only and must not be used for
training, curriculum construction, threshold selection, or model selection.

The weakly supervised training data are not distributed in this repository.
Place the training JSON in this directory or pass its path with `--train-file`.
