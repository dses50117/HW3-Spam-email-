#!/usr/bin/env python3
"""
Preprocess email/SMS text data for spam classification.

This script normalizes raw text data by applying deterministic transformations:
- Lowercase conversion
- Whitespace trimming and collapsing
- URL/email/phone replacement with tokens
- Number replacement with tokens
- Punctuation stripping (preserving intra-word apostrophes/hyphens)
- Optional stopword removal
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import re

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from utils.text_processing import normalize_text


def preprocess_emails(
    input_path,
    output_path,
    label_col=None,
    text_col=None,
    label_col_index=None,
    text_col_index=None,
    no_header=False,
    output_text_col='text_clean',
    save_step_columns=False,
    steps_out_dir=None
):
    """
    Preprocess email/SMS dataset.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        label_col: Name of the label column
        text_col: Name of the text column
        label_col_index: Index of the label column
        text_col_index: Index of the text column
        no_header: Whether the input CSV has no header
        output_text_col: Name for the final cleaned text column
        save_step_columns: Whether to save intermediate step columns
        steps_out_dir: Directory to save intermediate step files
    """
    print(f"Reading data from {input_path}...")
    
    header = None if no_header else 0
    df = pd.read_csv(input_path, header=header)
    
    if no_header:
        df.columns = [f'col_{i}' for i in range(df.shape[1])]
        if label_col_index is not None:
            label_col = f'col_{label_col_index}'
        if text_col_index is not None:
            text_col = f'col_{text_col_index}'

    label_series = df[label_col]
    text_series = df[text_col].fillna('')
    
    print(f"Preprocessing {len(df)} samples...")
    
    output_df = pd.DataFrame()
    output_df[label_col] = label_series
    
    if save_step_columns:
        output_df[text_col] = text_series
        output_df['text_lower'] = text_series.apply(lambda x: str(x).lower())
        
        def mask_contacts(text):
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)
            text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
            text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '<PHONE>', text)
            text = re.sub(r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b', '<PHONE>', text)
            return text
        output_df['text_contacts_masked'] = text_series.apply(mask_contacts)
        
        output_df['text_numbers'] = text_series.apply(lambda x: re.sub(r'\b\d+\b', '<NUM>', str(x)))
        
        def strip_punctuation(text):
            text = re.sub(r'\s+([^\w\s<>])\s+', ' ', text)
            text = re.sub(r'^[^\w\s<>]+|[^\w\s<>]+$', '', text)
            return text
        output_df['text_stripped'] = text_series.apply(strip_punctuation)
        
        output_df['text_whitespace'] = text_series.apply(lambda x: ' '.join(str(x).split()))
        
        def remove_stopwords_only(text):
            stopwords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with'
            }
            words = str(text).split()
            return ' '.join([w for w in words if w not in stopwords])
        output_df['text_stopwords_removed'] = text_series.apply(remove_stopwords_only)
        
        if steps_out_dir:
            steps_path = Path(steps_out_dir)
            steps_path.mkdir(parents=True, exist_ok=True)
            for col in output_df.columns:
                if col not in [label_col, text_col]:
                    step_df = pd.DataFrame({label_col: label_series, col: output_df[col]})
                    step_df.to_csv(steps_path / f'{col}.csv', index=False)
                    print(f"Saved step column to {steps_path / f'{col}.csv'}")

    output_df[output_text_col] = text_series.apply(lambda x: normalize_text(str(x)))
    
    final_df = pd.DataFrame({label_col: label_series, output_text_col: output_df[output_text_col]})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if save_step_columns:
        output_df.to_csv(output_path, index=False)
    else:
        final_df.to_csv(output_path, index=False)

    print(f"Saved cleaned data to {output_path}")
    print(f"Shape: {output_df.shape if save_step_columns else final_df.shape}")
    print(f"\nSample cleaned texts:")
    print(output_df.head(3) if save_step_columns else final_df.head(3))


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess email/SMS text data for spam classification'
    )
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    parser.add_argument('--no-header', action='store_true', help='Set if the input CSV has no header')
    parser.add_argument('--label-col', help='Name of label column')
    parser.add_argument('--text-col', help='Name of text column')
    parser.add_argument('--label-col-index', type=int, help='Index of label column (if no header)')
    parser.add_argument('--text-col-index', type=int, help='Index of text column (if no header)')
    parser.add_argument('--output-text-col', default='text_clean', help='Name for the final cleaned text column')
    parser.add_argument('--save-step-columns', action='store_true', help='Save all intermediate cleaning steps as columns')
    parser.add_argument('--steps-out-dir', help='Directory to save intermediate step columns as separate files')
    
    args = parser.parse_args()
    
    preprocess_emails(
        input_path=args.input,
        output_path=args.output,
        label_col=args.label_col,
        text_col=args.text_col,
        label_col_index=args.label_col_index,
        text_col_index=args.text_col_index,
        no_header=args.no_header,
        output_text_col=args.output_text_col,
        save_step_columns=args.save_step_columns,
        steps_out_dir=args.steps_out_dir
    )


if __name__ == '__main__':
    main()