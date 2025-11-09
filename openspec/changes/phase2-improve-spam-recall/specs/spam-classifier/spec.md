# Spam Classifier Capability - Phase 2 Deltas

## MODIFIED Requirements

### Requirement: Model Training
The system SHALL train a Logistic Regression classifier using TF-IDF features on preprocessed data with configurable hyperparameters for recall optimization.

#### Scenario: Train baseline model
- **WHEN** user runs `python scripts/train_spam_classifier.py --input datasets/processed/sms_spam_clean.csv --label-col col_0 --text-col text_clean`
- **THEN** the script SHALL train a model and save artifacts to `models/` directory

#### Scenario: Training metrics displayed
- **WHEN** training completes
- **THEN** the script SHALL display Accuracy, Precision, Recall, and F1 scores on the test set

#### Scenario: Model persistence
- **WHEN** training completes successfully
- **THEN** the script SHALL save the trained model and vectorizer using joblib to `models/spam_classifier.pkl` and `models/tfidf_vectorizer.pkl`

#### Scenario: Train with class weighting
- **WHEN** user runs training with `--class-weight balanced`
- **THEN** the model SHALL apply inverse class frequency weighting to improve minority class recall

#### Scenario: Train with bigrams
- **WHEN** user runs training with `--ngram-range 1,2`
- **THEN** the TF-IDF vectorizer SHALL include both unigrams and bigrams as features

#### Scenario: Train with minimum document frequency filter
- **WHEN** user runs training with `--min-df 2`
- **THEN** the vectorizer SHALL ignore terms appearing in fewer than 2 documents

#### Scenario: Train with sublinear TF scaling
- **WHEN** user runs training with `--sublinear-tf`
- **THEN** the vectorizer SHALL apply sublinear term frequency scaling (1 + log(tf))

#### Scenario: Train with custom regularization
- **WHEN** user runs training with `--C 0.5`
- **THEN** the model SHALL use inverse regularization strength of 0.5

#### Scenario: Evaluate with custom threshold
- **WHEN** user runs training with `--eval-threshold 0.4`
- **THEN** the evaluation SHALL classify samples as spam if probability ≥ 0.4

## ADDED Requirements

### Requirement: High Recall Configuration
The system SHALL provide a recommended configuration that achieves spam recall ≥ 0.93.

#### Scenario: Recommended high-recall training
- **WHEN** user trains with recommended parameters: `--class-weight balanced --ngram-range 1,2 --min-df 2 --sublinear-tf --C 0.5 --eval-threshold 0.40`
- **THEN** the model SHALL achieve spam recall ≥ 0.93 on the test set

#### Scenario: Trade-off documentation
- **WHEN** user reads the README section on recall tuning
- **THEN** they SHALL find documented trade-offs between recall and precision with example metrics

### Requirement: Hyperparameter Flexibility
The system SHALL allow users to experiment with different hyperparameter combinations via CLI flags.

#### Scenario: Combined parameter usage
- **WHEN** user specifies multiple flags in a single command
- **THEN** all parameters SHALL be applied simultaneously to both vectorizer and model

#### Scenario: Default backward compatibility
- **WHEN** user runs training without new flags
- **THEN** the system SHALL use Phase 1 baseline defaults (no class weighting, unigrams only, etc.)
