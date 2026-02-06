import psycopg2

# PostgreSQL config
POSTGRES_HOST = "localhost"   # <-- changé
POSTGRES_PORT = 5432
POSTGRES_DB = "iris_db"
POSTGRES_USER = "arthur_mcwalter_sperger"
POSTGRES_PASSWORD = "securepassword123"

try:
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    cur = conn.cursor()
    cur.execute("SELECT * FROM iris LIMIT 5;")
    rows = cur.fetchall()
    print("✅ Connexion OK, 5 premières lignes :")
    for row in rows:
        print(row)
    cur.close()
    conn.close()
except Exception as e:
    print("❌ Erreur :", e)
