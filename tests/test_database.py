import pytest
from src.db.database import get_db_connection

def test_db_connection():
    try:
        conn = get_db_connection()
        assert conn is not None
        conn.close()
    except Exception as e:
        pytest.fail(f"Ошибка при подключении к базе данных: {e}")

def test_insert_and_delete_ticket():
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO ticket_predictions (ticket_id, title, description, predicted_type)
            VALUES (%s, %s, %s, %s) RETURNING id
        """, ("test_id", "Test Title", "Test Description", "Test Category"))
        conn.commit()
    except Exception as e:
        pytest.fail(f"Ошибка при вставке записи: {e}")

    try:
        cursor.execute("SELECT * FROM ticket_predictions WHERE ticket_id = %s", ("test_id",))
        record = cursor.fetchone()
        assert record is not None
    except Exception as e:
        pytest.fail(f"Ошибка при проверке записи: {e}")

    try:
        cursor.execute("DELETE FROM ticket_predictions WHERE ticket_id = %s", ("test_id",))
        conn.commit()
    except Exception as e:
        pytest.fail(f"Ошибка при удалении записи: {e}")

    cursor.close()
    conn.close()