import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('omw-1.4')

def clean_text(text):
    """Metni küçük harfe çevirir, sayıları maskeler ve noktalamayı kaldırır."""
    text = text.lower()
    text = re.sub(r'\d+', 'num', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text):
    """Metni kelimelerine ayırır (Tokenization) ve köklerine indirger (Lemmatization)."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) 
    
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words
    ]
    return " ".join(cleaned_tokens)