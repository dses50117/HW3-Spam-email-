# Implementation Tasks - Phase 1

## 1. Project Setup
- [ ] 1.1 Create directory structure (`datasets/`, `datasets/processed/`, `models/`, `scripts/`)
- [ ] 1.2 Create `requirements.txt` with dependencies: scikit-learn, pandas, numpy, joblib, scipy
- [ ] 1.3 Download SMS spam dataset to `datasets/sms_spam_no_header.csv`
- [ ] 1.4 Create `.gitignore` to exclude `models/` and `datasets/`

## 2. Data Preprocessing Script
- [ ] 2.1 Create `scripts/preprocess_emails.py` with CLI arguments
- [ ] 2.2 Implement text normalization:
  - [ ] Lowercase conversion
  - [ ] Whitespace trimming and collapsing
  - [ ] URL/email/phone replacement with tokens (`<URL>`, `<EMAIL>`, `<PHONE>`)
  - [ ] Number replacement with `<NUM>` token
  - [ ] Punctuation stripping (preserve intra-word apostrophes/hyphens)
- [ ] 2.3 Add optional stopword removal flag (`--remove-stopwords`, off by default)
- [ ] 2.4 Save cleaned CSV to `datasets/processed/`
- [ ] 2.5 Test preprocessing with sample data

## 3. Training Pipeline
- [ ] 3.1 Create `scripts/train_spam_classifier.py` with CLI arguments
- [ ] 3.2 Implement data loading from cleaned CSV
- [ ] 3.3 Implement train/test split (80/20)
- [ ] 3.4 Build TF-IDF vectorizer
- [ ] 3.5 Train Logistic Regression model
- [ ] 3.6 Calculate and display metrics (Accuracy, Precision, Recall, F1)
- [ ] 3.7 Save model and vectorizer to `models/` using joblib
- [ ] 3.8 Test training pipeline end-to-end

## 4. Prediction CLI
- [ ] 4.1 Create `scripts/predict_spam.py` with CLI arguments
- [ ] 4.2 Implement model and vectorizer loading
- [ ] 4.3 Implement single-text prediction with probability output
- [ ] 4.4 Add batch prediction from file option
- [ ] 4.5 Test prediction with sample texts

## 5. Documentation
- [ ] 5.1 Create README with setup instructions
- [ ] 5.2 Document preprocessing command with examples
- [ ] 5.3 Document training command with examples
- [ ] 5.4 Document prediction command with examples
- [ ] 5.5 Document expected metrics for baseline
