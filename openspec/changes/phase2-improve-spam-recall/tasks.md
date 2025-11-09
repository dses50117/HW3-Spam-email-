# Implementation Tasks - Phase 2

## 1. Extend Training Script
- [ ] 1.1 Add `--class-weight` flag (options: None, 'balanced')
- [ ] 1.2 Add `--ngram-range` flag (e.g., '1,2' for unigrams+bigrams)
- [ ] 1.3 Add `--min-df` flag (minimum document frequency filter)
- [ ] 1.4 Add `--sublinear-tf` flag (use sublinear TF scaling)
- [ ] 1.5 Add `--C` flag (inverse regularization strength)
- [ ] 1.6 Add `--eval-threshold` flag (decision threshold for classification)

## 2. Update TF-IDF Vectorizer
- [ ] 2.1 Modify vectorizer to accept ngram_range parameter
- [ ] 2.2 Modify vectorizer to accept min_df parameter
- [ ] 2.3 Modify vectorizer to accept sublinear_tf parameter
- [ ] 2.4 Test vectorizer with new parameters

## 3. Update Logistic Regression
- [ ] 3.1 Add class_weight parameter to model initialization
- [ ] 3.2 Add C parameter to model initialization
- [ ] 3.3 Implement custom threshold evaluation logic
- [ ] 3.4 Test model with new parameters

## 4. Hyperparameter Tuning
- [ ] 4.1 Run grid search to find optimal parameters for recall
- [ ] 4.2 Document recommended configuration achieving Recall ≥ 0.93
- [ ] 4.3 Validate metrics on test set
- [ ] 4.4 Record Precision, Recall, F1, and Accuracy

## 5. Documentation Updates
- [ ] 5.1 Update README with new training flags
- [ ] 5.2 Document recommended configuration for high recall
- [ ] 5.3 Explain recall/precision trade-offs
- [ ] 5.4 Provide example commands with new flags
- [ ] 5.5 Document observed metrics (target: Recall ≥ 0.93)
