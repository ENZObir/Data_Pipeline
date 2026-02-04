import pandas as pd

from models.Models import df


def load_data():
    df = pd.read_csv('api/iris.csv')
    print(df.head())
    return df

def basic_cleanning(df):
    df.dropna()
    df.drop_duplicates()
    print(f"Shape après nettoyage : {df.shape}")
    return df 

def split_flower(df):
    df_virginica = df[df['species'] == 'virginica']
    df_versicolor = df[df['species'] == 'versicolor']
    df_setosa = df[df['species'] == 'setosa']    
    print(f"Virginica : {len(df_virginica)} lignes")
    print(f"Versicolor : {len(df_versicolor)} lignes")
    print(f"Setosa : {len(df_setosa)} lignes")
    return df_virginica, df_versicolor, df_setosa



    

def main():
    df = load_data()
    basic_cleanning(df)
    df_virginica, df_versicolor, df_setosa = split_flower(df)
    df.to_csv('./api/clean_iris.csv', encoding='UTF-8', index=False)
    df_virginica.to_csv('./api/virignica.csv', encoding='UTF-8', index=False)
    df_versicolor.to_csv('./api/versicolor.csv', encoding='UTF-8', index=False)
    df_setosa.to_csv('./api/setosa.csv', encoding='UTF-8', index=False)

if __name__ == "__main__":
    main()




