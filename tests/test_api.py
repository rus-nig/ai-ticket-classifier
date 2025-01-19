import pytest
import requests
import os
import io

BASE_URL = "http://127.0.0.1:5050"

# -------------------- /status --------------------
def test_status_success():
    response = requests.get(f"{BASE_URL}/status")
    assert response.status_code == 200
    assert response.json() == {"status": "API работает"}

# -------------------- /categorize --------------------
def test_categorize_success():
    payload = {"description": "This is a test ticket", "id": "123", "title": "Test Title"}
    response = requests.post(f"{BASE_URL}/categorize", json=payload)
    assert response.status_code == 200
    assert "predicted_type" in response.json()

def test_categorize_missing_description():
    payload = {"id": "123", "title": "Test Title"}
    response = requests.post(f"{BASE_URL}/categorize", json=payload)
    assert response.status_code == 400
    assert response.json() == {"error": "Отсутствует описание тикета"}

def test_categorize_model_not_loaded():
    os.rename('models/ticket_classifier.pkl', 'models/ticket_classifier_backup.pkl')
    os.rename('models/tfidf_vectorizer.pkl', 'models/tfidf_vectorizer_backup.pkl')
    payload = {"description": "This is a test ticket", "id": "123", "title": "Test Title"}
    response = requests.post(f"{BASE_URL}/categorize", json=payload)
    assert response.status_code == 500
    assert response.json() == {"error": "Модель не загружена"}

    os.rename('models/ticket_classifier_backup.pkl', 'models/ticket_classifier.pkl')
    os.rename('models/tfidf_vectorizer_backup.pkl', 'models/tfidf_vectorizer.pkl')

# -------------------- /data --------------------
def test_manage_data_get_empty():
    if os.path.exists('data/tickets.csv'):
        os.rename('data/tickets.csv', 'data/tickets_backup.csv')
    response = requests.get(f"{BASE_URL}/data")
    assert response.status_code == 404
    assert response.json() == {"error": "Файл данных отсутствует"}

    if os.path.exists('data/tickets_backup.csv'):
        os.rename('data/tickets_backup.csv', 'data/tickets.csv')

def test_manage_data_post_success():
    csv_data = "Description;Type\nTest ticket 1;Bug\nTest ticket 2;Feature\n"
    files = {'file': ('test_data.csv', io.BytesIO(csv_data.encode('utf-8')), 'text/csv')}
    response = requests.post(f"{BASE_URL}/data", files=files)
    assert response.status_code == 200
    assert response.json() == {"message": "Файл успешно загружен"}

def test_manage_data_post_missing_file():
    response = requests.post(f"{BASE_URL}/data")
    assert response.status_code == 400
    assert response.json() == {"error": "Файл не предоставлен"}

def test_manage_data_post_invalid_format():
    txt_data = "This is not a CSV file."
    files = {'file': ('invalid_file.txt', io.BytesIO(txt_data.encode('utf-8')), 'text/plain')}
    response = requests.post(f"{BASE_URL}/data", files=files)
    assert response.status_code == 400
    assert response.json() == {"error": "Файл должен быть в формате CSV"}

# -------------------- /model-files --------------------
def test_model_files_post_success():
    model_data = b"Test model binary content"
    vectorizer_data = b"Test vectorizer binary content"
    files = {
        'model': ('test_model.pkl', io.BytesIO(model_data), 'application/octet-stream'),
        'vectorizer': ('test_vectorizer.pkl', io.BytesIO(vectorizer_data), 'application/octet-stream')
    }
    response = requests.post(f"{BASE_URL}/model-files", files=files)
    assert response.status_code == 200
    assert response.json() == {"message": "Модель и векторизатор успешно загружены"}

def test_model_files_post_missing_files():
    response = requests.post(f"{BASE_URL}/model-files")
    assert response.status_code == 400
    assert response.json() == {"error": "Оба файла (модель и векторизатор) должны быть предоставлены"}

def test_model_files_get_model_success():
    response = requests.get(f"{BASE_URL}/model-files?type=model", stream=True)
    assert response.status_code == 200

def test_model_files_get_vectorizer_success():
    response = requests.get(f"{BASE_URL}/model-files?type=vectorizer", stream=True)
    assert response.status_code == 200

def test_model_files_get_invalid_type():
    response = requests.get(f"{BASE_URL}/model-files?type=invalid", stream=True)
    assert response.status_code == 400
    assert response.json() == {"error": "Некорректный параметр запроса. Укажите 'type=model' или 'type=vectorizer'."}

# -------------------- /train-model --------------------
def test_train_model_without_db():
    payload = {"load_from_db": False}
    response = requests.post(f"{BASE_URL}/train-model", json=payload)
    assert response.status_code in [200, 500]

def test_train_model_with_db():
    payload = {"load_from_db": True}
    response = requests.post(f"{BASE_URL}/train-model", json=payload)
    assert response.status_code in [200, 500]

# -------------------- /tickets --------------------
def test_get_tickets_empty():
    response = requests.get(f"{BASE_URL}/tickets")
    assert response.status_code == 200
    assert isinstance(response.json().get("tickets"), list)

def test_delete_ticket_not_found():
    response = requests.delete(f"{BASE_URL}/tickets/99999")
    assert response.status_code == 404
    assert response.json() == {"message": "Тикет с ID 99999 не найден"}

def test_update_ticket_not_found():
    payload = {"title": "Updated Title"}
    response = requests.put(f"{BASE_URL}/tickets/99999", json=payload)
    assert response.status_code == 404
    assert response.json() == {"message": "Тикет с ID 99999 не найден"}