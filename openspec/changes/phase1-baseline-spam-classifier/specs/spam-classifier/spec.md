# Spam Classifier Capability

## ADDED Requirements

### Requirement: Data Preprocessing
The system SHALL provide a preprocessing script that normalizes raw email/SMS text data deterministically.

#### Scenario: Preprocess SMS spam dataset
- **WHEN** user runs `python scripts/preprocess_emails.py --input datasets/sms_spam_no_header.csv --output datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col col_1`
- **THEN** the script SHALL create a cleaned CSV at the output path with normalized text

#### Scenario: Text normalization applied
- **WHEN** preprocessing runs on text "Check out http://example.com! Call 555-1234 NOW!!!"
- **THEN** the output SHALL be "check out <URL> call <PHONE> now"

#### Scenario: Configurable columns
- **WHEN** user specifies `--label-col label --text-col text`
- **THEN** the script SHALL read from those column names in the input CSV

### Requirement: Model Training
The system SHALL train a Logistic Regression classifier using TF-IDF features on preprocessed data.

#### Scenario: Train baseline model
- **WHEN** user runs `python scripts/train_spam_classifier.py --input datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col text_clean`
- **THEN** the script SHALL train a model and save artifacts to `models/` directory

#### Scenario: Training metrics displayed
- **WHEN** training completes
- **THEN** the script SHALL display Accuracy, Precision, Recall, and F1 scores on the test set

#### Scenario: Model persistence
- **WHEN** training completes successfully
- **THEN** the script SHALL save the trained model and vectorizer using joblib to `models/spam_classifier.pkl` and `models/tfidf_vectorizer.pkl`

### Requirement: Spam Prediction
The system SHALL provide a CLI for predicting whether text is spam using the trained model.

#### Scenario: Single text prediction
- **WHEN** user runs `python scripts/predict_spam.py --text "Congratulations! You've won a free prize!"`
- **THEN** the script SHALL output the prediction label (spam/ham) and probability

#### Scenario: Load trained artifacts
- **WHEN** prediction script starts
- **THEN** it SHALL load the model from `models/spam_classifier.pkl` and vectorizer from `models/tfidf_vectorizer.pkl`

#### Scenario: Batch prediction from file
- **WHEN** user runs `python scripts/predict_spam.py --input-file test_messages.txt`
- **THEN** the script SHALL predict labels for all lines in the file and output results

### Requirement: Baseline Performance
The system SHALL achieve minimum baseline performance metrics on the SMS spam dataset.

#### Scenario: Baseline accuracy threshold
- **WHEN** training on the SMS spam dataset with default parameters
- **THEN** the test accuracy SHALL be at least 0.95

#### Scenario: Baseline recall threshold
- **WHEN** evaluating on the test set
- **THEN** the spam recall SHALL be at least 0.85

#### Scenario: Reproducible results
- **WHEN** the same preprocessing and training commands are run twice
- **THEN** the metrics SHALL be identical (deterministic with fixed random seed)
