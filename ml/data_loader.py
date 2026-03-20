import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data():
    df=pd.read_csv('data/spam.csv',encoding='latin-1')
    print("Initial Data:")
    print(df.head())
    print(df.columns.tolist())

    print("\nAfter cleaning the columns:\n")
    df=df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"])

    print(df.columns.tolist())
    print(df.head())

    print(f"\n Before removing the duplicate rows {df.shape[0]} rows\n")
    print('\n after removing the duplicate rows:\n')
    df=df.drop_duplicates()
    print(df.shape[0])

    print("\n Before removing the duplicate columns:\n")
    print(df.columns.tolist())
    df = df.loc[:, ~df.columns.duplicated()]
    print("\n After removing the duplicate columns:\n")
    print(df.columns.tolist())

    print("\n Checkign null values:\n")
    print(df.isnull().sum())

    print("\n After removing the null values:\n")
    df = df.dropna()
    print(df.shape[0])

    print("\nRenaming columns:\n")
    df = df.rename(columns={'v1':'label','v2':'message'})
    print(df.columns.tolist())

    le=LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    print("\nAfter encoding the labels:\n")
    print(df.head())

    return df
