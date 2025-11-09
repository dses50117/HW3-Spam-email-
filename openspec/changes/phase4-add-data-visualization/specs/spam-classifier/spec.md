# Spam Classifier Capability - Phase 4 Deltas

## ADDED Requirements

### Requirement: Static Visualization Generation
The system SHALL provide a CLI tool to generate static visualization reports for data analysis and model performance.

#### Scenario: Generate all visualizations
- **WHEN** user runs `python scripts/visualize_spam.py --input datasets/processed/sms_spam_clean.csv --model models/spam_classifier.pkl --vectorizer models/tfidf_vectorizer.pkl --output reports/visualizations/`
- **THEN** the script SHALL generate and save all visualization files (class distribution, token frequencies, confusion matrix, ROC curve, PR curve, threshold sweep)

#### Scenario: Class distribution chart
- **WHEN** visualization script processes the dataset
- **THEN** it SHALL generate a bar chart showing the count of spam vs ham messages

#### Scenario: Top tokens visualization
- **WHEN** visualization script analyzes token frequencies
- **THEN** it SHALL generate separate frequency bar charts for top N tokens in spam and ham classes

#### Scenario: Confusion matrix generation
- **WHEN** visualization script evaluates the model on test data
- **THEN** it SHALL generate a confusion matrix heatmap showing true positives, true negatives, false positives, and false negatives

#### Scenario: ROC curve generation
- **WHEN** visualization script evaluates model performance
- **THEN** it SHALL generate an ROC curve plot with AUC score displayed

#### Scenario: Precision-Recall curve generation
- **WHEN** visualization script evaluates model performance
- **THEN** it SHALL generate a Precision-Recall curve plot with average precision score

#### Scenario: Threshold sweep analysis
- **WHEN** user requests threshold analysis
- **THEN** the script SHALL generate a plot showing Precision, Recall, and F1 scores across threshold values from 0.0 to 1.0

### Requirement: Interactive Dashboard
The system SHALL provide a Streamlit web application for interactive data exploration and live inference.

#### Scenario: Launch Streamlit dashboard
- **WHEN** user runs `streamlit run app/streamlit_app.py`
- **THEN** a web browser SHALL open displaying the interactive dashboard

#### Scenario: View data overview
- **WHEN** user navigates to the data overview section
- **THEN** they SHALL see class distribution chart and dataset statistics

#### Scenario: Explore token patterns
- **WHEN** user navigates to the token analysis section
- **THEN** they SHALL see top token frequency charts for spam and ham separately

#### Scenario: Review model performance
- **WHEN** user navigates to the model performance section
- **THEN** they SHALL see confusion matrix, ROC curve, and Precision-Recall curve

#### Scenario: Interactive threshold analysis
- **WHEN** user views the threshold analysis section
- **THEN** they SHALL see a plot of metrics vs threshold with an interactive slider

#### Scenario: Adjust threshold slider
- **WHEN** user moves the threshold slider
- **THEN** the plot SHALL highlight the current threshold and display corresponding Precision, Recall, and F1 values

#### Scenario: Live spam prediction
- **WHEN** user enters text in the inference text box and clicks predict
- **THEN** the dashboard SHALL display the prediction (spam/ham) and probability score

#### Scenario: Dynamic threshold inference
- **WHEN** user adjusts the threshold slider during inference
- **THEN** the prediction label SHALL update in real-time based on the new threshold

### Requirement: Reproducible Reports
The system SHALL save all generated visualizations to a designated output directory for documentation and presentation.

#### Scenario: Output directory creation
- **WHEN** visualization script runs
- **THEN** it SHALL create the `reports/visualizations/` directory if it doesn't exist

#### Scenario: Figure file naming
- **WHEN** visualizations are saved
- **THEN** files SHALL have descriptive names (e.g., `class_distribution.png`, `confusion_matrix.png`, `roc_curve.png`, `top_spam_tokens.png`, `top_ham_tokens.png`, `threshold_sweep.png`)

#### Scenario: Consistent image format
- **WHEN** figures are saved
- **THEN** they SHALL use a consistent format (PNG with 300 DPI for publication quality)

### Requirement: Visualization Dependencies
The system SHALL include all necessary libraries for generating and displaying visualizations.

#### Scenario: Visualization libraries installed
- **WHEN** user installs project dependencies
- **THEN** matplotlib, seaborn, and streamlit SHALL be installed from requirements.txt

#### Scenario: Import verification
- **WHEN** visualization scripts import required libraries
- **THEN** no import errors SHALL occur for matplotlib, seaborn, or streamlit
