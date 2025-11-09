#!/usr/bin/env python3
"""
Predict spam/ham for text messages using a trained model.

This script loads a trained spam classifier and makes predictions on new text,
either from a single string or a batch from a CSV file.
"""

import argparse
import joblib
import sys
import pandas as pd
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.text_processing import normalize_text


def load_model(model_path, vectorizer_path):
    """Load trained model and vectorizer."""
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}", file=sys.stderr)
        print("Please train a model first using train_spam_classifier.py", file=sys.stderr)
        sys.exit(1)
    
    if not Path(vectorizer_path).exists():
        print(f"Error: Vectorizer file not found: {vectorizer_path}", file=sys.stderr)
        print("Please train a model first using train_spam_classifier.py", file=sys.stderr)
        sys.exit(1)
    
    model_data = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    return model_data, vectorizer


def predict_single(text, model_data, vectorizer):
    """
    Predict spam/ham for a single text.
    
    Returns:
        tuple: (label, probability, is_spam)
    """
    normalized_text = normalize_text(text)
    X = vectorizer.transform([normalized_text])
    
    model = model_data['model']
    label_mapping = model_data['label_mapping']
    threshold = model_data.get('threshold', 0.5)
    
    proba = model.predict_proba(X)[0]
    spam_proba = proba[1]
    
    is_spam = spam_proba >= threshold
    
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    predicted_label = reverse_mapping[1] if is_spam else reverse_mapping[0]
    
    return predicted_label, spam_proba, is_spam


def predict_batch(input_path, text_col, model_data, vectorizer, output_path=None):
    """
    Predict spam/ham for multiple texts from a CSV file.
    """
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if text_col not in df.columns:
        print(f"Error: Text column '{text_col}' not found in {input_path}", file=sys.stderr)
        sys.exit(1)

    texts = df[text_col].fillna('').tolist()
    print(f"Predicting {len(texts)} texts from {input_path}...")
    
    predictions = []
    for text in texts:
        label, proba, _ = predict_single(text, model_data, vectorizer)
        predictions.append({'prediction': label, 'spam_probability': proba})
    
    pred_df = pd.DataFrame(predictions)
    output_df = pd.concat([df, pred_df], axis=1)
    
    if output_path:
        output_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
    else:
        print("\nPredictions:")
        print(output_df.head())
    
    return output_df


def main():
    parser = argparse.ArgumentParser(
        description='Predict spam/ham for text messages'
    )
    
    # Model paths
    parser.add_argument('--model-path', default='models/spam_logreg_model.joblib', help='Path to trained model')
    parser.add_argument('--vectorizer-path', default='models/spam_tfidf_vectorizer.joblib', help='Path to TF-IDF vectorizer')
    
    # Input options
    parser.add_argument('--text', type=str, help='Single text to classify')
    parser.add_argument('--input', type=str, help='Path to input CSV file for batch prediction')
    parser.add_argument('--text-col', type=str, help='Text column name for batch prediction')
    
    # Output option
    parser.add_argument('--output', type=str, help='Path to save batch predictions (CSV)')
    
    args = parser.parse_args()
    
    if not args.text and not args.input:
        parser.error("Either --text or --input must be provided.")
    if args.input and not args.text_col:
        parser.error("--text-col is required when using --input.")

    print("Loading model...")
    model_data, vectorizer = load_model(args.model_path, args.vectorizer_path)
    
    if args.text:
        label, proba, is_spam = predict_single(args.text, model_data, vectorizer)
        emoji = "🚫 SPAM" if is_spam else "✓ HAM"
        print(f"\nPREDICTION: {emoji}")
        print(f"  Label: {label.upper()}")
        print(f"  Spam Probability: {proba:.4f}")
        
    elif args.input:
        predict_batch(args.input, args.text_col, model_data, vectorizer, args.output)


if __name__ == '__main__':
    main()