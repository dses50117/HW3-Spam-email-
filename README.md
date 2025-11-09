# Spam Email Classification Project

Machine learning project for spam email classification using TF-IDF and Logistic Regression (SVM-based).

## Project Overview

This project implements a spam classifier through four progressive phases:
- **Phase 1**: Baseline classifier (Accuracy ≥95%, Recall ≥85%) ✅
- **Phase 2**: Improved recall (Recall ≥93%)
- **Phase 3**: Balanced precision and recall (Precision ≥90%, Recall ≥93%)
- **Phase 4**: Data visualization and interactive dashboard

## Setup

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation

1. Clone or download this repository

2. Create and activate virtual environment:
```powershell
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

### Dataset

The project uses the SMS Spam Collection Dataset. Download it:
```powershell
# Already included in datasets/ directory
# Or download manually from:
# https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv
```

## Usage

### Phase 1: Baseline Classifier

#### Step 1: Preprocess Data

```powershell
python scripts/preprocess_emails.py `
    --input datasets/sms_spam_no_header.csv `
    --output datasets/processed/sms_spam_clean.csv `
    --label-col col_0 `
    --text-col col_1
```

**What it does:**
- Converts text to lowercase
- Replaces URLs, emails, phone numbers with tokens (`<URL>`, `<EMAIL>`, `<PHONE>`)
- Replaces numbers with `<NUM>` token
- Trims whitespace and removes punctuation
- Saves cleaned data to `datasets/processed/`

#### Step 2: Train Model

```powershell
python scripts/train_spam_classifier.py `
    --input datasets/processed/sms_spam_clean.csv `
    --label-col col_0 `
    --text-col text_clean
```

**Actual Results (Phase 1 Baseline):**
```
Accuracy:  0.9794  (Target: ≥0.95) ✅
Precision: 0.9846
Recall:    0.8591  (Target: ≥0.85) ✅
F1 Score:  0.9176
```

**Output:**
- Model saved to: `models/spam_classifier.pkl`
- Vectorizer saved to: `models/tfidf_vectorizer.pkl`

#### Step 3: Make Predictions

**Single text prediction:**
```powershell
python scripts/predict_spam.py `
    --text "Congratulations! You've won a FREE prize!"
```

**Batch prediction from file:**
```powershell
python scripts/predict_spam.py `
    --input-file test_messages.txt `
    --output-file predictions.csv
```

### Phase 2: Improve Recall (Recall ≥ 0.93)

Train with hyperparameters optimized for high recall:

```powershell
python scripts/train_spam_classifier.py `
    --input datasets/processed/sms_spam_clean.csv `
    --label-col col_0 `
    --text-col text_clean `
    --class-weight balanced `
    --ngram-range 1,2 `
    --min-df 2 `
    --sublinear-tf `
    --C 0.5 `
    --eval-threshold 0.40
```

**New Flags:**
- `--class-weight balanced`: Apply inverse class frequency weighting
- `--ngram-range 1,2`: Include unigrams + bigrams
- `--min-df 2`: Ignore terms appearing in <2 documents
- `--sublinear-tf`: Use sublinear term frequency scaling
- `--C 0.5`: Regularization strength (lower = more regularization)
- `--eval-threshold 0.40`: Lower threshold to catch more spam

**Actual Results (Phase 2):**
```
Accuracy:  0.9758
Precision: 0.8675
Recall:    0.9664  (Target: ≥0.93) ✅ (Improved!)
F1 Score:  0.9143
```

### Phase 3: Balance Precision and Recall

Train with parameters optimized for balanced performance:

```powershell
python scripts/train_spam_classifier.py `
    --input datasets/processed/sms_spam_clean.csv `
    --label-col col_0 `
    --text-col text_clean `
    --class-weight balanced `
    --ngram-range 1,2 `
    --min-df 2 `
    --sublinear-tf `
    --C 2.0 `
    --eval-threshold 0.50
```

**Key Changes from Phase 2:**
- `--C 2.0`: Less regularization (was 0.5)
- `--eval-threshold 0.50`: Higher threshold for better precision (was 0.40)

**Actual Results (Phase 3 - Balanced):**
```
Accuracy:  0.9874  (Target: ≥0.98) ✅
Precision: 0.9592  (Target: ≥0.90) ✅ (Improved!)
Recall:    0.9463  (Target: ≥0.93) ✅ (Maintained!)
F1 Score:  0.9527  (Target: ≥0.94) ✅
```

### Phase 4: Visualization ✅

Generate static visualizations:
```powershell
python scripts/visualize_spam.py `
    --input datasets/processed/sms_spam_clean.csv `
    --model models/spam_classifier_phase3.pkl `
    --vectorizer models/tfidf_vectorizer_phase3.pkl `
    --output reports/visualizations/
```

**Generated Visualizations:**
- `class_distribution.png` - Bar chart of spam vs ham counts
- `top_tokens.png` - Top 20 spam and ham indicator tokens
- `confusion_matrix.png` - Heatmap of classification results
- `roc_curve.png` - ROC curve with AUC score
- `precision_recall_curve.png` - PR curve with average precision
- `threshold_sweep.png` - Metrics vs threshold plot
- `threshold_sweep.csv` - Detailed threshold analysis table

Launch interactive Streamlit dashboard:
```powershell
streamlit run app/streamlit_app.py
```

**Dashboard Features:**
- 📊 Data Overview: Dataset statistics and class distribution
- 🔤 Token Analysis: Top spam/ham indicator words
- 📈 Model Performance: Confusion matrix and metrics
- 🎯 Threshold Analysis: Interactive threshold tuning
- 🔮 Live Prediction: Real-time spam detection with adjustable threshold

## Project Structure

```
HW3/
├── datasets/
│   ├── sms_spam_no_header.csv        # Raw dataset
│   └── processed/
│       └── sms_spam_clean.csv        # Cleaned dataset
├── models/
│   ├── spam_classifier.pkl           # Trained model
│   └── tfidf_vectorizer.pkl          # TF-IDF vectorizer
├── scripts/
│   ├── preprocess_emails.py          # Data preprocessing
│   ├── train_spam_classifier.py      # Model training
│   ├── predict_spam.py               # Inference
│   └── visualize_spam.py             # Visualization (Phase 4)
├── app/
│   └── streamlit_app.py              # Interactive dashboard (Phase 4)
├── reports/
│   └── visualizations/               # Generated plots
├── openspec/                         # OpenSpec proposals
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Performance Comparison

| Phase | Accuracy | Precision | Recall | F1     | Focus        | Status |
|-------|----------|-----------|--------|--------|--------------|--------|
| 1     | 0.9794   | 0.9846    | 0.8591 | 0.9176 | Baseline     | ✅ Complete |
| 2     | 0.9758   | 0.8675    | 0.9664 | 0.9143 | High Recall  | ✅ Complete |
| 3     | 0.9874   | 0.9592    | 0.9463 | 0.9527 | Balanced     | ✅ Complete |
| 4     | (Same as Phase 3 + visualizations & dashboard) | Visualization | ✅ Complete |

## Understanding Metrics

- **Accuracy**: Overall correctness (correct predictions / total predictions)
- **Precision**: Of predicted spam, how many are actually spam (minimize false positives)
- **Recall**: Of actual spam, how many we caught (minimize false negatives)
- **F1 Score**: Harmonic mean of precision and recall (balanced metric)

### When to Use Each Configuration

- **Phase 1 (Baseline)**: Quick validation, initial testing
- **Phase 2 (High Recall)**: Critical to catch all spam, false positives acceptable
- **Phase 3 (Balanced)**: Production use, best overall performance
- **Custom threshold**: Adjust `--eval-threshold` based on your needs
  - Lower (0.3-0.4): Catch more spam (higher recall, lower precision)
  - Higher (0.6-0.7): Fewer false alarms (higher precision, lower recall)

## Troubleshooting

### Issue: ModuleNotFoundError
```
Solution: Ensure virtual environment is activated and dependencies installed
(.venv) PS> pip install -r requirements.txt
```

### Issue: Model file not found
```
Solution: Train the model first
python scripts/train_spam_classifier.py --input datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col text_clean
```

### Issue: Dataset not found
```
Solution: Run preprocessing first
python scripts/preprocess_emails.py --input datasets/sms_spam_no_header.csv --output datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col col_1
```

## Development

This project follows the OpenSpec spec-driven development workflow. See `openspec/` directory for detailed change proposals and specifications.

### Current Phase Status
- ✅ Phase 1: Complete (Baseline achieved: Accuracy 97.94%, Recall 85.91%)
- ✅ Phase 2: Complete (High recall achieved: Recall 96.64%)
- ✅ Phase 3: Complete (Balanced achieved: Precision 95.92%, Recall 94.63%)
- ✅ Phase 4: Complete (All visualizations and interactive dashboard implemented)

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Dataset: SMS Spam Collection from "Hands-On Artificial Intelligence for Cybersecurity"
- Built with: scikit-learn, pandas, numpy, matplotlib, seaborn, streamlit
