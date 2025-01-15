import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname="ticket_classifier",
            user="postgres",
            password="",
            host="localhost",
            port="5432"
        )
        return conn
    except Exception as e:
        print("Ошибка подключения к БД:", e)
        return None