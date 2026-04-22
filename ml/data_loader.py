import pandas as pd
import os


def load_data():
    # get the project root so this works no matter where we run from
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'spam.csv')

    # latin-1 encoding because the csv has special chars that break utf-8
    df = pd.read_csv(file_path, encoding='latin-1')

    # csv has 3 junk columns with no data, drop them
    df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

    # rename v1/v2 to something readable
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})

    # using map instead of LabelEncoder so we control which is 0 and which is 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    df = df.dropna()
    df = df.drop_duplicates()

    return df
