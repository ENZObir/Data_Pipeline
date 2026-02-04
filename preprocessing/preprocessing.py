from json import load
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

def load_data():
    df = pd.read_csv('api/iris.csv')
    print(df.head())
    return df


def basic_cleanning(df):
    df.dropna()
    df.drop_duplicates()
    print(df.shape)
    return df 

def split_flower(df):
    df_virginica = df[df['species'] == 'virginica']
    df_versicolor = df[df['species'] == 'versicolor']
    df_setosa = df[df['species'] == 'setosa']

    print(df_virginica.head())
    return df_virginica, df_versicolor, df_setosa


def main():
    df = load_data()
    basic_cleanning(df)
    split_flower(df)

if __name__ == "__main__":
    main()




