# Spam Email Classification Project - OpenSpec Proposals

## Project Overview
Machine learning coursework project to classify spam emails using Logistic Regression (SVM-based) with iterative performance improvements.

**Dataset**: SMS Spam Collection from GitHub
**Goal**: Build a baseline classifier and progressively improve recall, then precision, then add visualization

---

## Phase Structure

### ✅ Phase 1: Baseline Spam Classifier
**Location**: `openspec/changes/phase1-baseline-spam-classifier/`
**Goal**: Establish working baseline with Accuracy ≥ 0.95, Recall ≥ 0.85

**Deliverables**:
- Data preprocessing script with text normalization
- Training pipeline (TF-IDF + Logistic Regression)
- Prediction CLI for inference
- Model artifacts saved to `models/`

**Key Files**:
- `proposal.md` - Why and what changes
- `tasks.md` - 5 sections, 27 implementation tasks
- `specs/spam-classifier/spec.md` - Requirements with scenarios

---

### ✅ Phase 2: Improve Spam Recall
**Location**: `openspec/changes/phase2-improve-spam-recall/`
**Goal**: Achieve Recall ≥ 0.93 (catch more spam)

**Deliverables**:
- Extended training script with hyperparameter flags
- Class weighting, bigrams, min_df, sublinear TF
- Configurable evaluation threshold
- Documentation of recall/precision trade-offs

**New CLI Flags**: `--class-weight`, `--ngram-range`, `--min-df`, `--sublinear-tf`, `--C`, `--eval-threshold`

**Key Files**:
- `proposal.md`
- `tasks.md` - 5 sections, 15 implementation tasks
- `specs/spam-classifier/spec.md` - Delta changes (MODIFIED + ADDED requirements)

---

### ✅ Phase 3: Improve Spam Precision
**Location**: `openspec/changes/phase3-improve-spam-precision/`
**Goal**: Balance metrics (Precision ≥ 0.90, Recall ≥ 0.93)

**Deliverables**:
- Fine-tuned hyperparameters for balanced performance
- Recommended configuration: C=2.0, threshold=0.50
- Comparison table: Phase 1 vs 2 vs 3
- Parameter selection guidance

**Target Metrics**: Accuracy ≥ 0.98, Precision ≥ 0.92, Recall ≥ 0.96, F1 ≥ 0.94

**Key Files**:
- `proposal.md`
- `tasks.md` - 5 sections, 14 implementation tasks
- `specs/spam-classifier/spec.md` - Delta changes (MODIFIED + ADDED requirements)

---

### ✅ Phase 4: Add Data Visualization
**Location**: `openspec/changes/phase4-add-data-visualization/`
**Goal**: Create comprehensive visual reports and interactive dashboard

**Deliverables**:
- Visualization CLI (`scripts/visualize_spam.py`)
- Static plots: class distribution, token frequencies, confusion matrix, ROC, PR curves, threshold sweep
- Interactive Streamlit dashboard (`app/streamlit_app.py`)
- Live inference with adjustable threshold slider
- Reports saved to `reports/visualizations/`

**New Dependencies**: matplotlib, seaborn, streamlit

**Key Files**:
- `proposal.md`
- `tasks.md` - 6 sections, 30 implementation tasks
- `specs/spam-classifier/spec.md` - Delta changes (all ADDED requirements)

---

## Next Steps

### 1. Review and Validate Proposals
```powershell
# Validate each phase (requires openspec CLI)
openspec validate phase1-baseline-spam-classifier --strict
openspec validate phase2-improve-spam-recall --strict
openspec validate phase3-improve-spam-precision --strict
openspec validate phase4-add-data-visualization --strict
```

### 2. Implementation Order
Work through phases sequentially:
1. Start with Phase 1 (baseline)
2. Get approval before implementing
3. Complete all tasks in `tasks.md`
4. Archive after validation
5. Move to next phase

### 3. View All Proposals
```powershell
# List active changes
openspec list

# View specific proposal
openspec show phase1-baseline-spam-classifier
```

---

## Project Context
Updated `openspec/project.md` with:
- Purpose: ML coursework for spam classification
- Tech stack: Python, scikit-learn, pandas, matplotlib, streamlit
- Code style: PEP 8, snake_case files, 4-space indentation
- Architecture: Script-based CLI tools with separation of concerns
- Domain context: Spam classification, ML metrics, dataset details
- Constraints: Local execution, CPU-only, <5min training, reproducible

---

## Quick Commands Reference

### Phase 1
```powershell
# Preprocess
python scripts/preprocess_emails.py --input datasets/sms_spam_no_header.csv --output datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col col_1

# Train
python scripts/train_spam_classifier.py --input datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col text_clean

# Predict
python scripts/predict_spam.py --text "Win a free prize!"
```

### Phase 2 (High Recall)
```powershell
python scripts/train_spam_classifier.py \
  --input datasets/processed/sms_spam_clean.csv \
  --label-col col_0 --text-col text_clean \
  --class-weight balanced \
  --ngram-range 1,2 \
  --min-df 2 \
  --sublinear-tf \
  --C 0.5 \
  --eval-threshold 0.40
```

### Phase 3 (Balanced)
```powershell
python scripts/train_spam_classifier.py \
  --input datasets/processed/sms_spam_clean.csv \
  --label-col col_0 --text-col text_clean \
  --class-weight balanced \
  --ngram-range 1,2 \
  --min-df 2 \
  --sublinear-tf \
  --C 2.0 \
  --eval-threshold 0.50
```

### Phase 4 (Visualization)
```powershell
# Generate static visualizations
python scripts/visualize_spam.py --input datasets/processed/sms_spam_clean.csv --model models/spam_classifier.pkl --vectorizer models/tfidf_vectorizer.pkl --output reports/visualizations/

# Launch interactive dashboard
streamlit run app/streamlit_app.py
```

---

## Success Criteria

| Phase | Accuracy | Precision | Recall | F1    |
|-------|----------|-----------|--------|-------|
| 1     | ≥ 0.95   | -         | ≥ 0.85 | -     |
| 2     | -        | -         | ≥ 0.93 | -     |
| 3     | ≥ 0.98   | ≥ 0.90    | ≥ 0.93 | ≥ 0.94|
| 4     | (Same as Phase 3 + visualizations) |

---

**Status**: All 4 phase proposals created ✅
**Ready for**: Review and approval before implementation
