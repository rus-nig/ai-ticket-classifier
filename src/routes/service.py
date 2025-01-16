from flask import Flask, request, jsonify
from src.db.database import get_db_connection

import pickle
import csv
import os
import pandas as pd
import shutil

app = Flask(__name__)

# Пути к файлам
DATA_PATH = 'data/tickets.csv'
MODEL_PATH = 'models/ticket_classifier.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

# Глобальные переменные для модели и векторизатора
model, vectorizer = None, None

# ==================== Утилиты ====================

def export_tickets_to_csv():
    """
    Функция для экспорта данных из базы данных в tickets.csv.
    """
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT ticket_id, description, predicted_type FROM ticket_predictions")
    tickets_from_db = cursor.fetchall()

    # Если файл tickets.csv отсутствует, создаём его с корректными колонками
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(['id', 'title', 'Type', 'Description'])

    with open(DATA_PATH, mode='r', newline='') as csv_file:
        existing_data = list(csv.reader(csv_file, delimiter=';'))
        existing_ids = {row[0] for row in existing_data[1:]}

    # Преобразование данных из базы в формат CSV
    new_tickets = []
    for ticket in tickets_from_db:
        ticket_id, description, predicted_type = ticket

        new_row = [
            ticket_id,         # id
            "",                # title
            predicted_type,    # Type
            description        # Description
        ]

        if str(ticket_id) not in existing_ids:
            new_tickets.append(new_row)

    if new_tickets:
        with open(DATA_PATH, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerows(new_tickets)
        print(f"{len(new_tickets)} новых записей добавлено в {DATA_PATH}.")
    else:
        print("Новых записей для добавления нет.")

    cursor.close()
    connection.close()

def load_model_and_vectorizer():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return True
    return False

# Загрузка модели при старте сервера
load_model_and_vectorizer()

# ==================== Эндпоинты ====================

@app.route('/status', methods=['GET'])
def status():
    """Проверка работы API."""
    return jsonify({
        "status": "API работает"
    })

@app.route('/categorize', methods=['POST'])
def categorize():
    """Категоризация тикета и сохранение результата в БД."""
    if model is None or vectorizer is None:
        return jsonify({"error": "Модель не загружена"}), 500

    data = request.get_json()
    description = data.get('description', '')
    ticket_id = data.get('id')

    if not description:
        return jsonify({"error": "Отсутствует описание тикета"}), 400

    # Предсказание категории
    vectorized_description = vectorizer.transform([description])
    prediction = model.predict(vectorized_description)[0]

    # Сохранение в БД
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            query = """
                INSERT INTO ticket_predictions (ticket_id, description, predicted_type)
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, (ticket_id, description, prediction))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Ошибка сохранения в БД:", e)
        return jsonify({"error": "Ошибка сохранения данных в БД"}), 500

    return jsonify({
        "description": description,
        "predicted_type": prediction,
        "ticket_id": ticket_id
    }), 200

@app.route('/data', methods=['POST', 'GET'])
def manage_data():
    """Управление данными (загрузка/сохранение)."""
    if request.method == 'POST':
        file = request.files.get('file')
        
        if not file:
            return jsonify({"error": "Файл не предоставлен"}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Файл должен быть в формате CSV"}), 400

        temp_path = os.path.join('data', 'temp.csv')
        file.save(temp_path)

        try:
            data = pd.read_csv(temp_path, delimiter=';', usecols=['Description', 'Type'])
            required_columns = ['Description', 'Type']
            for col in required_columns:
                if col not in data.columns:
                    os.remove(temp_path)
                    return jsonify({"error": f"Отсутствует колонка '{col}' в CSV"}), 400

            shutil.copy(temp_path, DATA_PATH)
            os.remove(temp_path)
            return jsonify({"message": "Файл успешно загружен"}), 200

        except Exception as e:
            os.remove(temp_path)
            return jsonify({"error": f"Ошибка при обработке файла: {str(e)}"}), 400

    if request.method == 'GET':
        if not os.path.exists(DATA_PATH):
            return jsonify({"error": "Файл данных отсутствует"}), 404

        data = pd.read_csv(DATA_PATH, delimiter=';', on_bad_lines='skip').to_dict(orient='records')
        return jsonify({"data": data}), 200
    
@app.route('/load-model', methods=['POST'])
def load_model():
    """Загрузка пользовательской модели и векторизатора."""
    model_file = request.files.get('model')
    vectorizer_file = request.files.get('vectorizer')

    if not model_file or not vectorizer_file:
        return jsonify({"error": "Оба файла (модель и векторизатор) должны быть предоставлены"}), 400

    try:
        model_file.save(MODEL_PATH)
        vectorizer_file.save(VECTORIZER_PATH)

        load_model_and_vectorizer()

        return jsonify({"message": "Модель и векторизатор успешно загружены"}), 200

    except Exception as e:
        return jsonify({"error": f"Ошибка при загрузке модели: {str(e)}"}), 500
    
@app.route('/export-tickets', methods=['POST'])
def export_tickets():
    """
    Тестовый временный эндпоинт для экспорта данных из базы данных в tickets.csv.
    """
    export_tickets_to_csv()
    return jsonify({"message": "Данные успешно экспортированы."}), 200
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)