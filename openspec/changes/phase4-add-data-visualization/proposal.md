# Change: Phase 4 - Add Data Visualization

## Why
We want clear, reproducible visual reports for the spam classifier to aid understanding and presentation. Visualizations will show data distribution, token patterns, model performance metrics, and provide interactive exploration capabilities.

## What Changes
- Add a visualization CLI `scripts/visualize_spam.py` to generate static reports
- Create visualizations for:
  - Class distribution (spam vs ham) bar chart
  - Top token frequency charts (separate for spam and ham)
  - Confusion matrix on held-out test set
  - ROC curve and Precision-Recall curves
  - Threshold sweep analysis (Precision/Recall/F1 vs threshold)
- Add an interactive Streamlit dashboard `app/streamlit_app.py` for:
  - Interactive exploration of the same static visuals
  - Live inference text box with spam prediction
  - Adjustable threshold slider to see real-time impact
- Save generated figures and tables to `reports/visualizations/`
- Document usage and examples in README

## Impact
- **Affected specs**: Modifies `spam-classifier` capability (adds visualization)
- **Affected code**: 
  - New script: `scripts/visualize_spam.py`
  - New app: `app/streamlit_app.py`
  - New directory: `reports/visualizations/` (git-ignored if large)
  - Updated: `requirements.txt` with matplotlib, seaborn, streamlit
- **Dependencies**: Add `matplotlib`, `seaborn`, `streamlit`
- **Breaking changes**: None (new optional capability)

## Out of Scope
- Advanced dashboards (multi-page apps, authentication, persistent backends)
- SHAP or feature attribution beyond token frequency
- Real-time monitoring or alerting systems
- Cloud deployment of dashboard
