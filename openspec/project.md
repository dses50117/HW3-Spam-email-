# Project Context

## Purpose
Machine learning coursework project for spam email classification using Support Vector Machine (SVM) with Logistic Regression.

**Goals:**
- Build a reproducible spam email classifier using classical ML techniques
- Iteratively improve model performance (recall and precision)
- Demonstrate data preprocessing, feature engineering, and hyperparameter tuning
- Provide visualization and analysis tools for model evaluation
- Maintain clear documentation and OpenSpec-driven development process

**Key Objectives:**
1. Phase 1: Establish baseline classifier (Accuracy ≥ 0.95, Recall ≥ 0.85)
2. Phase 2: Improve spam recall (Recall ≥ 0.93)
3. Phase 3: Balance precision and recall (Precision ≥ 0.90, Recall ≥ 0.93)
4. Phase 4: Add comprehensive visualization and interactive dashboard

## Tech Stack
- **Language**: Python 3.8+
- **ML Library**: scikit-learn (TF-IDF vectorization, Logistic Regression)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Dashboard**: Streamlit (interactive web app)
- **Model Persistence**: joblib
- **Development**: Jupyter notebooks (optional for exploration)

## Project Conventions

### Code Style
- **Formatting**: Follow PEP 8 for Python code
- **Indentation**: 4 spaces (Python standard)
- **Naming conventions**:
  - Files: `snake_case.py` (e.g., `train_spam_classifier.py`)
  - Variables: `snake_case` (e.g., `test_accuracy`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD`)
  - Classes: `PascalCase` (e.g., `SpamClassifier`)
  - Functions: `snake_case` with descriptive verb-noun (e.g., `preprocess_text`, `train_model`)
- **Import organization**: 
  - Standard library imports first
  - Third-party imports second (pandas, numpy, sklearn)
  - Local imports last
  - Alphabetical within each group

### Architecture Patterns
- **Project structure**: Script-based CLI tools with optional notebooks
  ```
  datasets/          # Raw and processed data
  models/            # Trained model artifacts
  scripts/           # CLI tools for preprocessing, training, prediction, visualization
  app/               # Streamlit dashboard
  reports/           # Generated visualizations and analysis
  notebooks/         # (Optional) Exploratory analysis
  ```
- **Separation of concerns**: 
  - Preprocessing is a separate, deterministic stage
  - Training is configurable via CLI flags
  - Inference is decoupled from training
  - Visualization is standalone
- **Configuration**: CLI arguments for all parameters (no config files in Phase 1-4)
- **Reproducibility**: Fixed random seeds, deterministic preprocessing, versioned artifacts

### Testing Strategy
- **Manual validation**: Run scripts on test data and verify metrics
- **Reproducibility tests**: Same inputs produce same outputs
- **Metric thresholds**: Automated checks that accuracy/precision/recall meet targets
- **Visual inspection**: Generated plots and confusion matrices reviewed manually
- **Test data**: 80/20 train-test split with fixed random seed

### Git Workflow
- **Branch naming**: 
  - `phase1-baseline-spam-classifier`
  - `phase2-improve-spam-recall`
  - `phase3-improve-spam-precision`
  - `phase4-add-data-visualization`
- **Commit messages**: Descriptive, imperative mood (e.g., "Add preprocessing script with text normalization")
- **OpenSpec workflow**: Create proposal → Get approval → Implement → Archive
- **Phase-based development**: Each phase is a complete, working increment

## Domain Context

### Spam Classification
- **Spam**: Unsolicited, unwanted messages (often promotional or malicious)
- **Ham**: Legitimate, desired messages
- **False Positive**: Ham incorrectly classified as spam (user misses important message)
- **False Negative**: Spam incorrectly classified as ham (user receives unwanted message)

### Machine Learning Concepts
- **TF-IDF**: Term Frequency-Inverse Document Frequency (feature extraction from text)
- **Logistic Regression**: Linear classifier with probabilistic output (0-1)
- **Precision**: Of predicted spam, how many are actually spam (minimize false positives)
- **Recall**: Of actual spam, how many are detected (minimize false negatives)
- **F1 Score**: Harmonic mean of precision and recall (balanced metric)

### Use Case Trade-offs
- **High Recall Priority**: Email spam filters (better to catch all spam, some false positives acceptable)
- **High Precision Priority**: Banking fraud detection (false alarms are costly)
- **Balanced**: General production systems (Phase 3 target)

### Dataset
- **Source**: SMS Spam Collection Dataset
- **Format**: CSV with two columns (label, text)
- **Labels**: "spam" or "ham"
- **Size**: ~5,574 messages
- **Imbalance**: More ham than spam (class weighting needed)

## Important Constraints
- **Performance**: Training should complete in < 5 minutes on a laptop
- **Memory**: Dataset and model must fit in < 2GB RAM
- **Reproducibility**: All results must be deterministic with fixed random seed
- **Portability**: No GPU required, runs on CPU-only machines
- **Local execution**: No cloud services, all processing is local
- **Coursework deadline**: Phased delivery with clear milestones
- **Simplicity first**: Use scikit-learn only, no deep learning frameworks

## External Dependencies
- **Dataset source**: GitHub raw file URL (Hands-On AI for Cybersecurity repository)
  - URL: `https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv`
- **Python packages**: All from PyPI (pip installable)
  - Core: scikit-learn, pandas, numpy, joblib
  - Visualization: matplotlib, seaborn, streamlit
- **No external APIs** for training or inference
- **No authentication** or user management required
