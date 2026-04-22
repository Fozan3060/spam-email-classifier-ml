import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl

# fix ssl cert issue on mac for nltk downloads
ssl._create_default_https_context = ssl._create_unverified_context

# quiet=True so it doesn't spam the console every time
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

# create these once outside functions so they're not recreated for every row
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))  # set for O(1) lookup


def _lemmatize(words):
    # pos="v" treats words as verbs -> "running" becomes "run"
    return [wnl.lemmatize(word, pos="v") for word in words]


def _remove_stopwords(words):
    # filter out common words like "the", "is", "and" that don't help classification
    return [word for word in words if word not in stop_words]


def preprocess_text(df):
    # work on a copy so we don't modify the original dataframe
    df = df.copy()

    # chaining all text processing steps in order:
    # lowercase -> remove special chars -> strip whitespace -> split into words
    # -> remove stopwords -> lemmatize -> join back to string
    df['message'] = (
        df['message']
        .str.lower()
        .str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        .str.strip()
        .str.split()
        .apply(_remove_stopwords)
        .apply(_lemmatize)
        .apply(' '.join)
    )
    return df
