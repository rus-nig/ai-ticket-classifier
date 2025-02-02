import csv
import os
import pickle
import shutil

import pandas as pd
from flask import Flask, jsonify, request, send_file
from flasgger import Swagger
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.db.database import get_db_connection

app = Flask(__name__)

swagger_config = {
    "openapi": "3.0.0",
    "headers": {},
    "specs": [
        {
            "endpoint": "apispec",
            "route": "/apispec.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
    "version": "1.0.0",
}

swagger = Swagger(app, config=swagger_config)

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

    cursor.execute("SELECT ticket_id, title, description, predicted_type FROM ticket_predictions")
    tickets_from_db = cursor.fetchall()

    # Если файл tickets.csv отсутствует, создаём его с корректными колонками
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, mode='w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(['id', 'title', 'Type', 'Description'])

    with open(DATA_PATH, mode='r', encoding='utf-8') as csv_file:
        existing_data = list(csv.reader(csv_file, delimiter=';'))
        existing_ids = {row[0] for row in existing_data[1:]}

    # Преобразование данных из базы в формат CSV
    new_tickets = []
    for ticket in tickets_from_db:
        ticket_id, title, description, predicted_type = ticket

        new_row = [
            ticket_id or "",    # id
            title or "",        # title
            predicted_type,     # Type
            description         # Description
        ]

        if str(ticket_id) not in existing_ids:
            new_tickets.append(new_row)

    if new_tickets:
        with open(DATA_PATH, mode='a', encoding='utf-8', newline='') as csv_file:
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
    """
    Проверка работы API
    ---
    tags:
      - Health Check
    responses:
      200:
        description: API работает
        content:
          application/json:
            schema:
              type: object
              properties:
                status:
                  type: string
                  example: "API работает"
    """
    return jsonify({
        "status": "API работает"
    })

@app.route('/categorize', methods=['POST'])
def categorize():
    """
    Категоризация тикета
    ---
    tags:
      - Ticket Categorization
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              description:
                type: string
                example: "Текст тикета"
              id:
                type: string
                example: "12345"
              title:
                type: string
                example: "Название тикета"
    responses:
      200:
        description: Успешное предсказание
        content:
          application/json:
            schema:
              type: object
              properties:
                description:
                  type: string
                  example: "Текст тикета"
                predicted_type:
                  type: string
                  example: "Bug"
                ticket_id:
                  type: string
                  example: "12345"
                title:
                  type: string
                  example: "Название тикета"
      400:
        description: Ошибка в запросе
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Отсутствует описание тикета"
      500:
        description: Ошибка на сервере
    """
    if model is None or vectorizer is None:
        return jsonify({"error": "Модель не загружена"}), 500

    data = request.get_json()
    description = data.get('description', '')
    ticket_id = data.get('id')
    title = data.get('title', '')

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
                INSERT INTO ticket_predictions (ticket_id, title, description, predicted_type)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (ticket_id, title, description, prediction))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Ошибка сохранения в БД:", e)
        return jsonify({"error": "Ошибка сохранения данных в БД"}), 500

    return jsonify({
        "description": description,
        "predicted_type": prediction,
        "ticket_id": ticket_id,
        "title": title
    }), 200

@app.route('/data', methods=['POST', 'GET'])
def manage_data():
    """
    Управление данными
    ---
    tags:
      - Data Management
    requestBody:
      required: true
      content:
        multipart/form-data:
          schema:
            type: object
            required:
              - file
            properties:
              file:
                type: string
                format: binary
                description: CSV-файл с данными тикетов
    responses:
      200:
        description: Успешный ответ
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "Файл успешно загружен"
      400:
        description: Ошибка запроса
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Ошибка при загрузке файла"
      404:
        description: Файл данных отсутствует
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Файл данных отсутствует"
    """
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

@app.route('/model-files', methods=['GET', 'POST'])
def manage_model_files():
    """
    Управление файлами модели и векторизатора
    ---
    tags:
      - Model Management
    summary: Управление файлами модели и векторизатора
    description: Позволяет загружать и скачивать файлы модели и векторизатора.
    requestBody:
      required: true
      content:
        multipart/form-data:
          schema:
            type: object
            properties:
              model:
                type: string
                format: binary
                description: Файл модели.
              vectorizer:
                type: string
                format: binary
                description: Файл векторизатора.
    parameters:
      - in: query
        name: type
        required: false
        description: Тип запрашиваемого файла.
        schema:
          type: string
          enum:
            - model
            - vectorizer
    responses:
      200:
        description: Запрос выполнен успешно.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "Модель и векторизатор успешно загружены"
        application/octet-stream:
          schema:
            type: string
            format: binary
      400:
        description: Ошибка в запросе.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Некорректный параметр запроса. Укажите 'type=model' или 'type=vectorizer'."
      404:
        description: Файл не найден.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Файл модели отсутствует"
      500:
        description: Ошибка сервера.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Ошибка при загрузке файлов: <описание ошибки>"
    """
    if request.method == 'POST':
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
            return jsonify({"error": f"Ошибка при загрузке файлов: {str(e)}"}), 500

    elif request.method == 'GET':
        file_type = request.args.get('type')
        if file_type == 'model':
            if not os.path.exists(MODEL_PATH):
                return jsonify({"error": "Файл модели отсутствует"}), 404
            return send_file(os.path.abspath(MODEL_PATH), as_attachment=True)
        elif file_type == 'vectorizer':
            if not os.path.exists(VECTORIZER_PATH):
                return jsonify({"error": "Файл векторизатора отсутствует"}), 404
            return send_file(os.path.abspath(VECTORIZER_PATH), as_attachment=True)
        else:
            return jsonify({"error": "Некорректный параметр запроса. Укажите 'type=model' или 'type=vectorizer'."}), 400

@app.route('/train-model', methods=['POST'])
def train_model():
    """
    Обучение модели
    ---
    tags:
      - Model Training
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              load_from_db:
                type: boolean
                description: Загрузить дополнительные данные из базы перед обучением.
                example: true
    responses:
      200:
        description: Модель успешно обучена.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "Обучение завершено. Лучшая модель: Random Forest с точностью 0.9452"
                model_path:
                  type: string
                  example: "models/ticket_classifier.pkl"
                vectorizer_path:
                  type: string
                  example: "models/tfidf_vectorizer.pkl"
      404:
        description: Файл данных отсутствует.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Файл tickets.csv не найден"
      500:
        description: Ошибка обучения модели.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Ошибка обучения модели: <описание ошибки>"
    """
    # Проверяем, передал ли пользователь параметр для догрузки данных из БД
    data = request.get_json()
    load_from_db = data.get('load_from_db', False)

    if load_from_db:
        # Догружаем записи из базы данных в tickets.csv
        export_tickets_to_csv()
        db_message = "Данные из базы данных успешно добавлены в tickets.csv. "

    try:
        # Шаг 1: Загрузка данных
        if not os.path.exists(DATA_PATH):
            return jsonify({"error": "Файл tickets.csv не найден"}), 404
        data = pd.read_csv(DATA_PATH, delimiter=';')

        # Проверка наличия обязательных колонок
        required_columns = ['Description', 'Type']
        if not all(col in data.columns for col in required_columns):
            return jsonify({"error": f"Отсутствуют обязательные колонки: {required_columns}"}), 400

        # Шаг 2: Предобработка данных
        data = data.dropna(subset=['Description', 'Type'])
        data['Description'] = data['Description'].str.lower()
        data['Description'] = data['Description'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

        X = data['Description']
        y = data['Type']

        # Шаг 3: Разделение на обучающую и тестовую выборку
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Шаг 4: Векторизация текста
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Шаг 5: Обработка дисбаланса классов
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

        # Шаг 6: Обучение моделей
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(random_state=42, probability=True),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        param_grids = {
            "Logistic Regression": {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
            "KNN": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
            "Random Forest": {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]},
            "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
            "Gradient Boosting": {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2], 'max_depth': [3, 5]}
        }

        best_model_name = None
        best_model = None
        best_accuracy = 0

        for model_name, model in models.items():
            grid_search = GridSearchCV(model, param_grids[model_name], cv=3, scoring='accuracy')
            grid_search.fit(X_resampled, y_resampled)
            best_model_candidate = grid_search.best_estimator_

            # Оценка модели
            accuracy = accuracy_score(y_test, best_model_candidate.predict(X_test_tfidf))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
                best_model = best_model_candidate

        # Шаг 7: Сохранение лучшей модели и векторизатора
        with open(MODEL_PATH, 'wb') as model_file:
            pickle.dump(best_model, model_file)
            
        with open(VECTORIZER_PATH, 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)

        load_model_and_vectorizer()

        return jsonify({
            "message": f"{db_message}Обучение завершено. Лучшая модель: {best_model_name} с точностью {best_accuracy:.4f}",
            "model_path": MODEL_PATH,
            "vectorizer_path": VECTORIZER_PATH
        }), 200

    except Exception as e:
        return jsonify({"error": f"Ошибка обучения модели: {str(e)}"}), 500
    
@app.route('/tickets', methods=['GET'])
def get_tickets():
    """
    Получение всех записей
    ---
    tags:
      - Tickets
    summary: Получение всех записей тикетов
    description: Возвращает все записи тикетов из базы данных.
    responses:
      200:
        description: Записи успешно получены.
        content:
          application/json:
            schema:
              type: object
              properties:
                tickets:
                  type: array
                  items:
                    type: object
                    properties:
                      id:
                        type: integer
                        example: 1
                      title:
                        type: string
                        example: "Ошибка загрузки отчета"
                      description:
                        type: string
                        example: "Не удается загрузить отчет при выборе определенных фильтров"
                      predicted_type:
                        type: string
                        example: "Bug"
      500:
        description: Ошибка при получении данных из базы.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Ошибка при получении данных из базы"
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, description, predicted_type FROM ticket_predictions")
        tickets = cursor.fetchall()

        cursor.close()
        conn.close()

        tickets_list = [
            {"id": ticket[0], "title": ticket[1], "description": ticket[2], "predicted_type": ticket[3]}
            for ticket in tickets
        ]

        return jsonify({"tickets": tickets_list}), 200
    except Exception as e:
        print("Ошибка при получении данных из БД:", e)
        return jsonify({"error": "Ошибка при получении данных из базы"}), 500

@app.route('/tickets/<id>', methods=['DELETE'])
def delete_ticket(id):
    """
    Удаление записи
    ---
    tags:
      - Tickets
    summary: Удаление тикета
    description: Удаляет запись тикета из базы данных по указанному ID.
    parameters:
      - name: id
        in: path
        required: true
        schema:
          type: string
        description: Уникальный идентификатор тикета
        example: "123"
    responses:
      200:
        description: Запись успешно удалена.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "Тикет с ID 123 успешно удален"
      404:
        description: Запись не найдена.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "Тикет с ID 123 не найден"
      500:
        description: Ошибка при удалении записи.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Ошибка при удалении записи из базы"
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ticket_predictions WHERE id = %s", (id,))
        conn.commit()

        rows_deleted = cursor.rowcount
        cursor.close()
        conn.close()

        if rows_deleted == 0:
            return jsonify({"message": f"Тикет с ID {id} не найден"}), 404

        return jsonify({"message": f"Тикет с ID {id} успешно удален"}), 200
    except Exception as e:
        print("Ошибка при удалении записи из БД:", e)
        return jsonify({"error": "Ошибка при удалении записи из базы"}), 500

@app.route('/tickets/<id>', methods=['PUT'])
def update_ticket(id):
    """
    Обновление записи
    ---
    tags:
      - Tickets
    summary: Обновление тикета
    description: Обновляет поля существующей записи тикета в базе данных.
    parameters:
      - name: id
        in: path
        required: true
        schema:
          type: string
        description: Уникальный идентификатор тикета
        example: "123"
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              title:
                type: string
                description: Новый заголовок тикета
                example: "Обновленный заголовок"
              description:
                type: string
                description: Новое описание тикета
                example: "Обновленное описание тикета"
              predicted_type:
                type: string
                description: Новая категория тикета
                example: "Task"
    responses:
      200:
        description: Запись успешно обновлена.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "Тикет с ID 123 успешно обновлен"
      404:
        description: Запись не найдена.
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
                  example: "Тикет с ID 123 не найден"
      400:
        description: Отсутствуют данные для обновления.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Нет данных для обновления"
      500:
        description: Ошибка при обновлении записи.
        content:
          application/json:
            schema:
              type: object
              properties:
                error:
                  type: string
                  example: "Ошибка при обновлении записи в базе"
    """
    try:
        data = request.get_json()
        title = data.get('title')
        description = data.get('description')
        predicted_type = data.get('predicted_type')

        if not title and not description and not predicted_type:
            return jsonify({"error": "Нет данных для обновления"}), 400

        fields_to_update = []
        params = []
        if title:
            fields_to_update.append("title = %s")
            params.append(title)
        if description:
            fields_to_update.append("description = %s")
            params.append(description)
        if predicted_type:
            fields_to_update.append("predicted_type = %s")
            params.append(predicted_type)

        params.append(id)
        sql_query = f"UPDATE ticket_predictions SET {', '.join(fields_to_update)} WHERE id = %s"

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query, tuple(params))
        conn.commit()

        rows_updated = cursor.rowcount
        cursor.close()
        conn.close()

        if rows_updated == 0:
            return jsonify({"message": f"Тикет с ID {id} не найден"}), 404

        return jsonify({"message": f"Тикет с ID {id} успешно обновлен"}), 200
    except Exception as e:
        print("Ошибка при обновлении записи в БД:", e)
        return jsonify({"error": "Ошибка при обновлении записи в базе"}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)