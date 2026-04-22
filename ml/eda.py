import matplotlib.pyplot as plt
from data_loader import load_data


def perform_eda(df):
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # basic dataset overview
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}\n")
    print(f"Null Values:\n{df.isnull().sum()}\n")
    print(f"Duplicates: {df.duplicated().sum()}\n")

    # class distribution - important to check for imbalance
    print("Class Distribution:")
    print(df['label'].value_counts())
    print(f"\nPercentages:\n{df['label'].value_counts(normalize=True) * 100}\n")

    # look at actual messages to understand what we're working with
    print("Sample Spam Messages:")
    print(df[df['label'] == 1]['message'].head(), '\n')
    print("Sample Ham Messages:")
    print(df[df['label'] == 0]['message'].head(), '\n')

    # message length helps decide preprocessing strategy
    df['length'] = df['message'].str.len()
    print("Message Length Analysis:")
    print(df.groupby('label')['length'].describe())

    # visualize the class imbalance
    df['label'].value_counts().plot(kind='bar')
    plt.title('Spam vs Ham Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Ham', 'Spam'], rotation=0)
    plt.tight_layout()
    plt.show()

    return df


# only runs eda when this file is executed directly, not when imported
if __name__ == '__main__':
    perform_eda(load_data())
