from flask import Flask, request, jsonify

import pickle
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
    """Категоризация тикета."""
    if model is None or vectorizer is None:
        return jsonify({"error": "Модель не загружена"}), 500

    data = request.get_json()
    description = data.get('description', '')

    if not description:
        return jsonify({"error": "Отсутствует описание тикета"}), 400

    vectorized_description = vectorizer.transform([description])
    prediction = model.predict(vectorized_description)

    return jsonify({"description": description, "predicted_type": prediction[0]}), 200

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
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)