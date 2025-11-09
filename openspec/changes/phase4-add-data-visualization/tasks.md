# Implementation Tasks - Phase 4

## 1. Project Setup
- [ ] 1.1 Create `reports/visualizations/` directory
- [ ] 1.2 Add visualization dependencies to requirements.txt (matplotlib, seaborn, streamlit)
- [ ] 1.3 Create `app/` directory for Streamlit app
- [ ] 1.4 Update .gitignore for reports/ if needed

## 2. Visualization CLI Script
- [ ] 2.1 Create `scripts/visualize_spam.py` with CLI arguments
- [ ] 2.2 Implement class distribution bar chart
- [ ] 2.3 Implement top tokens frequency chart for spam class
- [ ] 2.4 Implement top tokens frequency chart for ham class
- [ ] 2.5 Implement confusion matrix heatmap
- [ ] 2.6 Implement ROC curve plot
- [ ] 2.7 Implement Precision-Recall curve plot
- [ ] 2.8 Implement threshold sweep analysis (table and plot)
- [ ] 2.9 Save all figures to `reports/visualizations/`
- [ ] 2.10 Test visualization script with trained model

## 3. Threshold Sweep Analysis
- [ ] 3.1 Implement threshold range (0.0 to 1.0 in steps)
- [ ] 3.2 Calculate Precision, Recall, F1 for each threshold
- [ ] 3.3 Generate threshold vs metrics line plot
- [ ] 3.4 Generate threshold comparison table
- [ ] 3.5 Highlight optimal thresholds

## 4. Streamlit Interactive Dashboard
- [ ] 4.1 Create `app/streamlit_app.py` with page layout
- [ ] 4.2 Add sidebar for navigation and controls
- [ ] 4.3 Implement data overview page (class distribution, dataset stats)
- [ ] 4.4 Implement token analysis page (frequency charts)
- [ ] 4.5 Implement model performance page (confusion matrix, ROC, PR curves)
- [ ] 4.6 Implement threshold analysis page (sweep plot with slider)
- [ ] 4.7 Implement live inference section with text input
- [ ] 4.8 Add threshold slider for inference (0.0 to 1.0)
- [ ] 4.9 Display prediction result (spam/ham) with probability
- [ ] 4.10 Test Streamlit app locally

## 5. Documentation
- [ ] 5.1 Document `visualize_spam.py` usage in README
- [ ] 5.2 Provide example commands for generating visualizations
- [ ] 5.3 Document Streamlit app launch command (`streamlit run app/streamlit_app.py`)
- [ ] 5.4 Add screenshots or descriptions of available visualizations
- [ ] 5.5 Document interpretation of metrics and charts

## 6. Integration Testing
- [ ] 6.1 Run visualization script with Phase 3 trained model
- [ ] 6.2 Verify all plots are generated and saved
- [ ] 6.3 Launch Streamlit app and test all pages
- [ ] 6.4 Test live inference with sample spam and ham texts
- [ ] 6.5 Verify threshold slider updates predictions correctly
