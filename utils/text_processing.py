import re

def normalize_text(text, remove_stopwords=False, keep_numbers=False):
    """
    Apply text normalization transformations. 
    
    Args:
        text: Input text string
        remove_stopwords: Whether to remove common stopwords
        keep_numbers: Whether to keep numbers (default: replace with <NUM>)
    
    Returns:
        Normalized text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with <URL> token
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '<URL>', text)
    
    # Replace emails with <EMAIL> token
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL>', text)
    
    # Replace phone numbers with <PHONE> token
    # Matches various phone formats: 555-1234, (555)1234, 555.1234, etc.
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '<PHONE>', text)
    text = re.sub(r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b', '<PHONE>', text)
    
    # Replace numbers with <NUM> token (unless keep_numbers is True)
    if not keep_numbers:
        text = re.sub(r'\b\d+\b', '<NUM>', text)
    
    # Trim and collapse whitespace
    text = ' '.join(text.split())
    
    # Strip surrounding punctuation, preserve intra-word apostrophes/hyphens
    # Remove punctuation at word boundaries
    text = re.sub(r'\s+([^\w\s<>])\s+', ' ', text)
    text = re.sub(r'^[^\w\s<>]+|[^\w\s<>]+$', '', text)
    
    # Optional: Remove stopwords
    if remove_stopwords:
        # Basic English stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with'
        }
        words = text.split()
        text = ' '.join([w for w in words if w not in stopwords])
    
    return text
