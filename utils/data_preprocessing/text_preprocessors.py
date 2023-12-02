import re
import string
import pandas as pd
from typing import Callable
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from utils.data_preprocessing.emo_unicode import EMOTICONS_EMO

#region CONSTANTS
_N_MOST_FREQ = 10
_STEMMER =  PorterStemmer()
_URL_PATTERN = re.compile(r"^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$")
_STOP_WORDS = set(stopwords.words("english"))
#endregion

#region UTILITIES
def _apply(data: pd.Series, func: Callable[[str], str]):
    return data.apply(func)
#endregion

#region LOWER CASE
def lower_case(text: pd.Series) -> pd.Series:
    return _apply(
        text, 
        lambda s: s.lower()
    )
#endregion

#region REMOVE PUNCTUATION
def remove_punctuations(text: pd.Series) -> pd.Series:
    # Refer: https://stackoverflow.com/a/266162
    return _apply(
        text, 
        lambda s: s.translate(str.maketrans('', '', string.punctuation))
    )
#endregion

#region REMOVE STOPWORDS
def remove_stopwords(text: pd.Series) -> pd.Series:
    return _apply(
        text, 
        lambda s: ' '.join([word for word in s.split() if word not in _STOP_WORDS])
    )
#endregion

#region REMOVE FREQUENT WORDS
def remove_n_most_freq_words(text: pd.Series) -> pd.Series:
    def get_n_freq_words(text):
        return dict(map(
            lambda x: x, 
            Counter(''.join(text).split()).most_common(_N_MOST_FREQ)
        ))
    most_freq_words = get_n_freq_words(text)
    print(f'N Most Freq Words: {most_freq_words}')
    
    return _apply(
        text, 
        lambda s: ' '.join([word for word in s.split() if word not in most_freq_words])
    )
#endregion

#region STEMMING
def apply_stem(text: pd.Series) -> pd.Series:
    return _apply(
        text, 
        lambda s: ' '.join([_STEMMER.stem(word) for word in s.split()])
    )
#endregion

#region CONVERT EMOTE
def convert_emote(text: pd.Series) -> pd.Series:
    return _apply(
        text, 
        lambda s: ' '.join([EMOTICONS_EMO.get(word, word) for word in s.split()])
    )
#endregion

#region REMOVE URLs
def remove_urls(text: pd.Series) -> pd.Series:
    return _apply(
        text, 
        lambda s: _URL_PATTERN.sub(r'', s)
    )
#endregion

#region PREPROCESSOR MAP
PREPROCESSOR: dict[str, Callable[[any], str]] = {
    'LOWER_CASE': lower_case,
    'REMOVE_PUNCTUATION': remove_punctuations,
    'REMOVE_STOPWORDS': remove_stopwords,
    'REMOVE_MOST_FREQ': remove_n_most_freq_words,
    'STEM': apply_stem,
    'CONVERT_EMOTE': convert_emote,
    'REMOVE_URLs': remove_urls
}
#endregion