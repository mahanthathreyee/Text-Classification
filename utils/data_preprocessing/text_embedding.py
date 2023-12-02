import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf(text: pd.Series):
    vectorizer = TfidfVectorizer()
    return vectorizer, vectorizer.fit_transform(text)