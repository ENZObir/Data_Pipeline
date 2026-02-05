import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values


ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
if ROOT_ENV.exists():
    load_dotenv(ROOT_ENV)


CSV_PATH = os.getenv("IRIS_CSV_PATH", "api/iris.csv")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_TABLE = os.getenv("POSTGRES_TABLE", "iris_data")


def load_data():
    df = pd.read_csv(CSV_PATH)
    print(df.head())
    return df


def basic_cleanning(df):
    cleaned = df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"Shape après nettoyage : {cleaned.shape}")
    return cleaned


def split_flower(df):
    df_virginica = df[df['species'] == 'virginica']
    df_versicolor = df[df['species'] == 'versicolor']
    df_setosa = df[df['species'] == 'setosa']
    print(f"Virginica : {len(df_virginica)} lignes")
    print(f"Versicolor : {len(df_versicolor)} lignes")
    print(f"Setosa : {len(df_setosa)} lignes")
    return df_virginica, df_versicolor, df_setosa


def export_csvs(cleaned, df_virginica, df_versicolor, df_setosa):
    os.makedirs('./data', exist_ok=True)
    cleaned.to_csv('./data/iris_clean.csv', encoding='UTF-8', index=False)
    df_virginica.to_csv('./data/virginica.csv', encoding='UTF-8', index=False)
    df_versicolor.to_csv('./data/versicolor.csv', encoding='UTF-8', index=False)
    df_setosa.to_csv('./data/setosa.csv', encoding='UTF-8', index=False)
    print("CSV générés dans ./data")


def load_into_database(df):
    records = df[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]].values.tolist()
    if not records:
        print("Aucune donnée à insérer")
        return

    with psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {POSTGRES_TABLE}")
            execute_values(
                cur,
                f"INSERT INTO {POSTGRES_TABLE} (sepal_length, sepal_width, petal_length, petal_width, species) VALUES %s",
                records,
            )
    print(f"{len(records)} lignes insérées dans {POSTGRES_TABLE}")


def main():
    df = load_data()
    cleaned = basic_cleanning(df)
    df_virginica, df_versicolor, df_setosa = split_flower(cleaned)
    export_csvs(cleaned, df_virginica, df_versicolor, df_setosa)
    load_into_database(cleaned)


if __name__ == "__main__":
    main()
