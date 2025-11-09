#!/usr/bin/env python3
"""
Train a spam classifier using TF-IDF and Logistic Regression.

This script trains a spam classification model on preprocessed text data.
Supports hyperparameter tuning for recall/precision optimization (Phase 2+).
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def train_spam_classifier(
    input_path,
    label_col,
    text_col,
    model_path='models/spam_classifier.pkl',
    vectorizer_path='models/tfidf_vectorizer.pkl',
    test_size=0.2,
    random_state=42,
    # Phase 2+ hyperparameters
    class_weight=None,
    ngram_range=(1, 1),
    min_df=1,
    sublinear_tf=False,
    C=1.0,
    eval_threshold=0.5
):
    """
    Train spam classifier with TF-IDF + Logistic Regression.
    
    Args:
        input_path: Path to preprocessed CSV
        label_col: Label column name
        text_col: Text column name
        model_path: Path to save trained model
        vectorizer_path: Path to save TF-IDF vectorizer
        test_size: Test set proportion
        random_state: Random seed for reproducibility
        class_weight: Class weighting strategy ('balanced' or None)
        ngram_range: N-gram range tuple (min_n, max_n)
        min_df: Minimum document frequency for terms
        sublinear_tf: Use sublinear TF scaling
        C: Inverse regularization strength
        eval_threshold: Decision threshold for classification
    """
    print("="*60)
    print("SPAM CLASSIFIER TRAINING")
    print("="*60)
    
    # Load data
    print(f"\n[1/6] Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} samples")
    
    # Check class distribution
    class_counts = df[label_col].value_counts()
    print(f"\n  Class distribution:")
    for label, count in class_counts.items():
        print(f"    {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # Prepare data
    X = df[text_col].fillna('').values
    y = df[label_col].values
    
    # Train-test split
    print(f"\n[2/6] Splitting data (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Create TF-IDF vectorizer
    print(f"\n[3/6] Creating TF-IDF vectorizer...")
    print(f"  ngram_range: {ngram_range}")
    print(f"  min_df: {min_df}")
    print(f"  sublinear_tf: {sublinear_tf}")
    
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        max_features=None
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Feature matrix shape: {X_train_tfidf.shape}")
    
    # Train Logistic Regression
    print(f"\n[4/6] Training Logistic Regression...")
    print(f"  class_weight: {class_weight}")
    print(f"  C: {C}")
    print(f"  eval_threshold: {eval_threshold}")
    
    model = LogisticRegression(
        class_weight=class_weight,
        C=C,
        max_iter=1000,
        random_state=random_state
    )
    
    model.fit(X_train_tfidf, y_train)
    print("  Training complete!")
    
    # Evaluate on test set
    print(f"\n[5/6] Evaluating on test set...")
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    
    # Apply custom threshold
    y_pred = (y_pred_proba >= eval_threshold).astype(int)
    
    # Convert string labels to binary if needed
    label_mapping = {val: idx for idx, val in enumerate(sorted(set(y_test)))}
    y_test_binary = np.array([label_mapping[label] for label in y_test])
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_binary, y_pred)
    precision = precision_score(y_test_binary, y_pred, pos_label=1)
    recall = recall_score(y_test_binary, y_pred, pos_label=1)
    f1 = f1_score(y_test_binary, y_pred, pos_label=1)
    
    print(f"\n  {'='*50}")
    print(f"  TEST SET METRICS (threshold={eval_threshold})")
    print(f"  {'='*50}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  {'='*50}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test_binary, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    [[TN={cm[0,0]:<4} FP={cm[0,1]:<4}]")
    print(f"     [FN={cm[1,0]:<4} TP={cm[1,1]:<4}]]")
    
    # Detailed classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_test_binary, y_pred, 
                                target_names=['ham', 'spam'], digits=4))
    
    # Save model and vectorizer
    print(f"\n[6/6] Saving model artifacts...")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save with metadata
    model_data = {
        'model': model,
        'label_mapping': label_mapping,
        'threshold': eval_threshold,
        'hyperparameters': {
            'class_weight': class_weight,
            'C': C,
            'ngram_range': ngram_range,
            'min_df': min_df,
            'sublinear_tf': sublinear_tf
        },
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }
    
    joblib.dump(model_data, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"  Model saved to: {model_path}")
    print(f"  Vectorizer saved to: {vectorizer_path}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}\n")
    
    return model, vectorizer, model_data


def main():
    parser = argparse.ArgumentParser(
        description='Train spam classifier with TF-IDF + Logistic Regression'
    )
    
    # Required arguments
    parser.add_argument('--input', required=True,
                        help='Path to preprocessed CSV file')
    parser.add_argument('--label-col', required=True,
                        help='Label column name')
    parser.add_argument('--text-col', required=True,
                        help='Text column name')
    
    # Output paths
    parser.add_argument('--model-path', default='models/spam_classifier.pkl',
                        help='Path to save trained model')
    parser.add_argument('--vectorizer-path', default='models/tfidf_vectorizer.pkl',
                        help='Path to save TF-IDF vectorizer')
    
    # Training parameters
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Phase 2+ hyperparameters
    parser.add_argument('--class-weight', choices=['balanced', 'none'], default='none',
                        help='Class weighting strategy (default: none)')
    parser.add_argument('--ngram-range', type=str, default='1,1',
                        help='N-gram range as "min,max" (default: 1,1)')
    parser.add_argument('--min-df', type=int, default=1,
                        help='Minimum document frequency (default: 1)')
    parser.add_argument('--sublinear-tf', action='store_true',
                        help='Use sublinear TF scaling')
    parser.add_argument('--C', type=float, default=1.0,
                        help='Inverse regularization strength (default: 1.0)')
    parser.add_argument('--eval-threshold', type=float, default=0.5,
                        help='Decision threshold for classification (default: 0.5)')
    
    args = parser.parse_args()
    
    # Parse ngram_range
    ngram_parts = args.ngram_range.split(',')
    ngram_range = (int(ngram_parts[0]), int(ngram_parts[1]))
    
    # Convert class_weight
    class_weight = args.class_weight if args.class_weight != 'none' else None
    
    # Train model
    train_spam_classifier(
        input_path=args.input,
        label_col=args.label_col,
        text_col=args.text_col,
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        test_size=args.test_size,
        random_state=args.random_state,
        class_weight=class_weight,
        ngram_range=ngram_range,
        min_df=args.min_df,
        sublinear_tf=args.sublinear_tf,
        C=args.C,
        eval_threshold=args.eval_threshold
    )


if __name__ == '__main__':
    main()
