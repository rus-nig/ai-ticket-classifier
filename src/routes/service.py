from flask import Flask, request, jsonify

import pickle
import os

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)