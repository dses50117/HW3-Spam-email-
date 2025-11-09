# Change: Phase 3 - Improve Spam Precision

## Why
After recall tuning in Phase 2, Recall is approximately 0.97 but Precision dropped to approximately 0.85. We need to regain Precision to ≥ 0.90 while maintaining Recall ≥ 0.93 to achieve a better balance and reduce false positives.

## What Changes
- Fine-tune evaluation threshold, TF-IDF parameters (min_df, n-grams), and regularization (C) to raise Precision
- Provide a recommended balanced configuration that achieves both Precision ≥ 0.90 and Recall ≥ 0.93
- Document trade-offs between precision and recall optimization
- Update README with guidance on parameter selection

## Impact
- **Affected specs**: Modifies `spam-classifier` capability
- **Affected code**: 
  - Documentation updates: README with precision optimization guidance
  - No new script changes (existing flags from Phase 2 are sufficient)
- **Breaking changes**: None (uses existing parameter flags)

## Out of Scope
- Dataset relabeling or external data collection
- Ensemble methods or model stacking
- Active learning or human-in-the-loop labeling
- Production deployment infrastructure
