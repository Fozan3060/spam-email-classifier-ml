from eda import perform_eda
from data_loader import load_data
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download("wordnet")
nltk.download("omw-1.4")
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lemmetizer(text):
    lemmetized_words=[]

    for word in text:
        lemmetized_words.append(wnl.lemmatize(word, pos="v"))
    
    return lemmetized_words

def removeStopwords(text):
    filtered_words=[]
    for word in text:
        if word not in stop_words:
            filtered_words.append(word)
    
    return filtered_words

def joinText(text):
    separator = ' '
    result_string = separator.join(text)
    return result_string

def text_processing(perform_eda):
    df = perform_eda
    df["message"] = df['message'].str.lower()
    print("Printing lower case df")
    df['message'] = df['message'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True).str.strip()
    df["message"] = df['message'].str.split()
    df['message'] = df['message'].apply(removeStopwords)
    df['message'] = df['message'].apply(lemmetizer)
    df['message'] = df['message'].apply(joinText)

    print(df.head())

    return df
    
text_processing(perform_eda(load_data()))


