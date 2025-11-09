# Change: Phase 2 - Improve Spam Recall

## Why
Recall on the spam class (~0.85) is below the target (≥ 0.93). We need to catch more spam messages while maintaining acceptable precision to reduce false negatives.

## What Changes
- Add training options to improve recall: class weighting, bi-gram features, min-df filtering, sublinear TF, and configurable evaluation threshold
- Extend `scripts/train_spam_classifier.py` with new CLI flags for hyperparameter tuning
- Retrain model with tuned settings optimized for recall
- Document metrics and trade-offs in README

## Impact
- **Affected specs**: Modifies `spam-classifier` capability
- **Affected code**: 
  - Modified script: `scripts/train_spam_classifier.py` (add new flags)
  - Updated documentation: README with recall/precision trade-offs
- **New flags**: `--class-weight`, `--ngram-range`, `--min-df`, `--sublinear-tf`, `--eval-threshold`, `--C`
- **Breaking changes**: None (backward compatible, new flags are optional)

## Out of Scope
- Deep learning models
- Dataset expansion or external data collection
- Real-time inference optimization
- Model serving infrastructure
