# Release and archived experiment correspondence

The portable implementation retains the active SciBERT span model, dataset/tokenization logic, decoding, strict/partial evaluator, confidence formula, type-wise quantiles, admission masks, loss normalization, and staged early-stopping behavior of the recorded complete method.

The archive used experiment-specific dynamic imports and absolute server paths. The release replaces those with local imports and command-line paths, removes unused soft-distillation and Monte Carlo experiment paths, and adds data-fingerprint checks. The manuscript's complete method uses three trained voters with deterministic scoring; MC dropout is not used in this release.

`source_manifest.json` lists the recovered source files and adaptation notes. The recorded run used 21,052 training instances, 2,632 updates per epoch, and 5/4/8 epochs in the three stages. Their best epochs were 2/1/5; the final selected stage-3 checkpoint produced strict F1 84.6418996553% and partial F1 89.5442359249%. Stage lengths arise from development performance, rather than being configured epoch limits.

The 188-text reference was used for development monitoring and model selection. It must not be described as an independent final test. Exact numerical reproduction requires the historical training view, tokenizer/encoder files, confidence inputs, seeds, GPU/software behavior, and selection procedure. Different API output, resource dictionaries, tokenizers, precision, or dependency versions may change results.

The released data-building scripts preserve recovered prompts and augmentation logic; they cannot reconstruct unrecorded PDF collection choices or guarantee identical generated samples. Source-document and semantic independence of the historical corpus has not been established by this release.

Historical optimization: AdamW, batch 8, learning rate 2e-5, weight decay 0.01, gradient clipping 1.0, BF16 training/scoring, FP32 final decoding. Each stage restarts its LR schedule after restoring the preceding stage-best model and Adam moments. Positive confidence weights remain fixed once a span is admitted; negatives remain active. The minimum two-epoch and patience-three rules apply per stage. The ten-epoch learning-rate horizon does not impose an epoch cap.

The public code package omits checkpoints and private corpora. The previously advertised external checkpoint has not been verified as the selected revised model, so this release does not claim that it reproduces the reported scores.
