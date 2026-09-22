import os
from urllib.parse import urlparse
import psycopg2
from app.core.config import settings

parsed = urlparse(settings.DATABASE_URL)
user = parsed.username or 'postgres'
password = parsed.password
host = parsed.hostname or 'localhost'
port = parsed.port or 5432
dbname = parsed.path.lstrip('/') or 'tempo_db'

print(f"Connecting to Postgres at {host}:{port} as {user} to check database '{dbname}'...")
conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname='postgres')
conn.autocommit = True
cur = conn.cursor()
cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
if not cur.fetchone():
    print(f"Creating database '{dbname}'...")
    cur.execute(f'CREATE DATABASE "{dbname}"')
    print(f"Database '{dbname}' created successfully.")
else:
    print(f"Database '{dbname}' already exists.")
cur.close()
conn.close()

# Now connect to tempo_db directly
conn_tempo = psycopg2.connect(host=host, port=port, user=user, password=password, dbname=dbname)
print("Successfully connected directly to tempo_db!")
conn_tempo.close()
