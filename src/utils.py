import string

def clean_text(text):
    """
    Lowercase the text and remove punctuation.
    """
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text
