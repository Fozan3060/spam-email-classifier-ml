from data_loader import load_data 
import matplotlib.pyplot as plt

def perform_eda(df):
    print("\n Performing EDA:\n")
    print(df.head())
    print(df.columns.tolist())
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1],'\n')
    print(df.info(),'\n')
    print(df.describe(),'\n')
    print(df.isnull().sum(),"\n")
    print(df.duplicated().sum(),'\n')
    print(df.nunique(),'\n')
    print(df['label'].value_counts(normalize=True)*100,'\n')

    print("Spam Emails \n:",df[df['label']==1]['message'].head(),'\n')
    print("Not Spam Emails \n:",df[df['label']==0]['message'].head(),'\n')

    df['length']=df['message'].str.len()
    print(df.head())
    print(df.groupby('label')['length'].describe())

    print("The bar char things:",df['label'].value_counts(),"\n")
    plt.title('Bar Chart of Emails')
    plt.xlabel('Number of emails')
    plt.ylabel('Emails Type')
    df['label'].value_counts().plot(kind='bar')
    plt.xticks([0, 1], ['Ham', 'Spam'])

    plt.show()

    return df

perform_eda(load_data())