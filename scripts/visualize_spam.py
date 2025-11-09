#!/usr/bin/env python3
"""
Generate visualizations for spam classifier analysis.

This script creates static visualizations including:
- Class distribution
- Top token frequencies (spam vs ham)
- Confusion matrix
- ROC and Precision-Recall curves
- Threshold sweep analysis
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_data(input_path):
    """Load dataset."""
    try:
        return pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

def load_model_artifacts(models_dir):
    """Load model and vectorizer."""
    model_path = Path(models_dir) / 'spam_logreg_model.joblib'
    vectorizer_path = Path(models_dir) / 'spam_tfidf_vectorizer.joblib'
    
    if not model_path.exists() or not vectorizer_path.exists():
        print(f"Error: Model artifacts not found in {models_dir}", file=sys.stderr)
        sys.exit(1)
        
    model_data = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model_data, vectorizer

def get_test_data(df, label_col, text_col, model_data, vectorizer):
    """Split data and get test set predictions."""
    X = df[text_col].fillna('').values
    y = df[label_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_test_tfidf = vectorizer.transform(X_test)
    
    label_mapping = model_data['label_mapping']
    y_test_binary = np.array([label_mapping.get(label, 0) for label in y_test])
    
    model = model_data['model']
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    return y_test_binary, y_pred_proba

def plot_class_distribution(df, label_col, output_dir):
    """Plot class distribution bar chart."""
    print("Generating class distribution chart...")
    plt.figure(figsize=(8, 6))
    counts = df[label_col].value_counts()
    colors = ['#2ecc71' if label == 'ham' else '#e74c3c' for label in counts.index]
    bars = plt.bar(counts.index, counts.values, color=colors, alpha=0.7, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}
({height/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Class Distribution: Spam vs Ham', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = Path(output_dir) / 'class_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_top_tokens(df, label_col, text_col, vectorizer, model_data, output_dir, top_n=20):
    """Plot top token frequencies for spam and ham."""
    print(f"Generating top {top_n} token frequency charts...")
    model = model_data['model']
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]
    
    top_spam_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_spam_tokens = feature_names[top_spam_idx]
    top_spam_weights = coefficients[top_spam_idx]
    
    top_ham_idx = np.argsort(coefficients)[:top_n]
    top_ham_tokens = feature_names[top_ham_idx]
    top_ham_weights = np.abs(coefficients[top_ham_idx])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.barh(range(top_n), top_spam_weights, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(top_n)); ax1.set_yticklabels(top_spam_tokens)
    ax1.set_xlabel('Weight (Spam Indicator)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Spam Indicators', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    ax2.barh(range(top_n), top_ham_weights, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(top_n)); ax2.set_yticklabels(top_ham_tokens)
    ax2.set_xlabel('Weight (Ham Indicator)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Top {top_n} Ham Indicators', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'top_tokens.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_confusion_matrix(y_test_binary, y_pred_proba, threshold, output_dir):
    """Plot confusion matrix heatmap."""
    print("Generating confusion matrix...")
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test_binary, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], annot_kws={'size': 16, 'weight': 'bold'})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix (threshold={threshold})', fontsize=14, fontweight='bold')
    tn, fp, fn, tp = cm.ravel()
    metrics_text = f'TN={tn}  FP={fp}\nFN={fn}  TP={tp}'
    plt.text(1, -0.15, metrics_text, ha='center', va='top', transform=plt.gca().transAxes, fontsize=10, family='monospace')
    plt.tight_layout()
    output_path = Path(output_dir) / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_roc_curve(y_test_binary, y_pred_proba, output_dir):
    """Plot ROC curve."""
    print("Generating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10); plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path = Path(output_dir) / 'roc_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_precision_recall_curve(y_test_binary, y_pred_proba, output_dir):
    """Plot Precision-Recall curve."""
    print("Generating Precision-Recall curve...")
    precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_proba)
    avg_precision = average_precision_score(y_test_binary, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#9b59b6', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10); plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path = Path(output_dir) / 'precision_recall_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

def plot_threshold_sweep(y_test_binary, y_pred_proba, output_dir):
    """Plot threshold sweep analysis."""
    print("Generating threshold sweep analysis...")
    thresholds = np.arange(0.0, 1.01, 0.05)
    precisions, recalls, f1_scores = [], [], []
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tp = np.sum((y_test_binary == 1) & (y_pred == 1))
        fp = np.sum((y_test_binary == 0) & (y_pred == 1))
        fn = np.sum((y_test_binary == 1) & (y_pred == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision); recalls.append(recall); f1_scores.append(f1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'o-', label='Precision', color='#3498db', linewidth=2)
    plt.plot(thresholds, recalls, 's-', label='Recall', color='#e74c3c', linewidth=2)
    plt.plot(thresholds, f1_scores, '^-', label='F1 Score', color='#2ecc71', linewidth=2)
    plt.xlabel('Decision Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Metrics vs Decision Threshold', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10); plt.grid(alpha=0.3)
    plt.xlim([0, 1]); plt.ylim([0, 1.05]); plt.tight_layout()
    
    output_path = Path(output_dir) / 'threshold_sweep.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    df_sweep = pd.DataFrame({'Threshold': thresholds, 'Precision': precisions, 'Recall': recalls, 'F1 Score': f1_scores})
    table_path = Path(output_dir) / 'threshold_sweep.csv'
    df_sweep.to_csv(table_path, index=False, float_format='%.4f')
    print(f"  Saved: {table_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for spam classifier')
    parser.add_argument('--input', required=True, help='Path to preprocessed CSV')
    parser.add_argument('--output', default='reports/visualizations/', help='Output directory for visualizations')
    parser.add_argument('--label-col', default='col_0', help='Label column name')
    parser.add_argument('--text-col', default='text_clean', help='Text column name')
    
    # Plot-specific flags
    parser.add_argument('--class-dist', action='store_true', help='Generate class distribution plot')
    parser.add_argument('--token-freq', action='store_true', help='Generate top token frequency plot')
    parser.add_argument('--confusion-matrix', action='store_true', help='Generate confusion matrix plot')
    parser.add_argument('--roc', action='store_true', help='Generate ROC curve plot')
    parser.add_argument('--pr', action='store_true', help='Generate Precision-Recall curve plot')
    parser.add_argument('--threshold-sweep', action='store_true', help='Generate threshold sweep plot and CSV')
    
    # Arguments for model-dependent plots
    parser.add_argument('--models-dir', default='models/', help='Directory containing model artifacts')
    parser.add_argument('--topn', type=int, default=20, help='Number of top tokens to display')

    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_data(args.input)
    
    model_plots = args.token_freq or args.confusion_matrix or args.roc or args.pr or args.threshold_sweep
    
    if model_plots:
        model_data, vectorizer = load_model_artifacts(args.models_dir)
        y_test_binary, y_pred_proba = get_test_data(df, args.label_col, args.text_col, model_data, vectorizer)
        threshold = model_data.get('threshold', 0.5)

    if args.class_dist:
        plot_class_distribution(df, args.label_col, output_dir)
        
    if args.token_freq:
        plot_top_tokens(df, args.label_col, args.text_col, vectorizer, model_data, output_dir, args.topn)
        
    if args.confusion_matrix:
        plot_confusion_matrix(y_test_binary, y_pred_proba, threshold, output_dir)
        
    if args.roc:
        plot_roc_curve(y_test_binary, y_pred_proba, output_dir)
        
    if args.pr:
        plot_precision_recall_curve(y_test_binary, y_pred_proba, output_dir)
        
    if args.threshold_sweep:
        plot_threshold_sweep(y_test_binary, y_pred_proba, output_dir)

    print("\nVisualization generation complete.")

if __name__ == '__main__':
    main()