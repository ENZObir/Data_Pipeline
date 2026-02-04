from json import load
import pandas as pd
import numpy as np
from sklearn.preprocessing import ColumnTransfromer
from sklearn.pipeline import Pipeline

def load_data():
    df = pd.read_csv('api/iris.csv')
    print(df.head())
    return df

df = load_data()


def basic_cleanning(df):
    df.dropna()
    df.drop_duplicates()
    print(df.shape)
    print()
    return df 


def main():
    df = load_data()
    df = basic_cleanning(df)





