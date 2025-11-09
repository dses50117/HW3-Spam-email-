# Change: Phase 1 - Baseline Spam Email Classifier

## Why
We need a machine learning solution to classify emails as spam or not spam for coursework. The goal is a simple, reproducible baseline that trains locally, evaluates with clear metrics, and provides a small CLI for inference.

## What Changes
- Add a new capability `spam-classifier` under OpenSpec
- Separate dataset preprocessing from model training with a standalone preprocessing step that reads raw CSVs, applies deterministic text normalization, and writes cleaned CSVs
- Implement a minimal training pipeline using scikit-learn (TF-IDF + Logistic Regression) over the cleaned dataset
- Persist trained artifacts under `models/` and expose a simple prediction CLI
- Document how to run preprocessing, training, and inference locally with Python
- Download and process SMS spam dataset from: `https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv`

## Impact
- **Affected specs**: New capability `spam-classifier`
- **Affected code**: 
  - New scripts: `scripts/preprocess_emails.py`, `scripts/train_spam_classifier.py`, `scripts/predict_spam.py`
  - New directory: `models/` for trained artifacts (git-ignored)
  - New directory: `datasets/` and `datasets/processed/` for data
- **Dependencies**: Add Python packages: `scikit-learn`, `pandas`, `numpy`, `joblib`, `scipy`
- **Breaking changes**: None (new capability)

## Out of Scope
- Cloud deployment or APIs
- Advanced model architectures (transformers, deep learning)
- Labeling or data collection tooling
- Real-time inference systems
