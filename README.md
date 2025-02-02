# AI Ticket Classifier

Инструмент для классификации тикетов на основе их описаний работающий при помощи машинного обучения. Этот проект предоставляет REST API для интеграции с системами управления проектами.

---

## Установка

### 1. Склонируйте репозиторий

### 2. Настройте виртуальное окружение
Создайте и активируйте виртуальное окружение:

Для **Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

Для **Linux/macOS**:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Установите зависимости
```bash
pip install -r requirements.txt
```

### 4. Настройте базу данных PostgreSQL
1. Убедитесь, что PostgreSQL установлен на вашем компьютере.

Для **Windows**:  
Скачайте установочный файл с официального сайта [PostgreSQL](https://www.postgresql.org/download/) и следуйте инструкциям.

Для **macOS**:
```bash
brew install postgresql
```

Для **Linux**:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

2. Запустите сервер PostgreSQL

Для **macOS**:
```bash
brew services start postgresql
```

Для **Linux**:
```bash
sudo service postgresql start
```

3. Создайте базу данных и таблицу для проекта:
Войдите в консоль PostgreSQL:
```bash
psql
```
Выполните команды:

```sql
CREATE DATABASE ticket_classifier;
\c ticket_classifier
CREATE TABLE ticket_predictions (
    id SERIAL PRIMARY KEY,
    ticket_id VARCHAR(50),
    description TEXT NOT NULL,
    predicted_type VARCHAR(50) NOT NULL,
    title TEXT
);
```
4. Настройте подключение в файле `src/db/database.py`, изменив параметры `dbname`, `user`, `password` и `host` при необходимости.

### 5. Запуск приложения

Используйте main.py для удобного запуска:

-	Для запуска API:

```bash
python main.py service
```
-	Для запуска веб-интерфейса:

```bash
python main.py app
```

-	Для запуска обоих:

```bash
python main.py
```

API-сервис будет доступен по адресу http://127.0.0.1:5050, а веб-интерфейс откроется в браузере.

---

## Возможности

- Предустановленная модель машинного обучения для классификации тикетов.
- Поддержка загрузки собственных наборов данных для обучения и тестирования.
- Поддержка загрузки и скачивания модели и векторизатора.
- Веб-интерфейс на основе Streamlit для взаимодействия с моделью и данными.
- REST API для интеграции с системами управления проектами.
- Эндпоинты для:
  - Загрузки/сохранения данных для обучения моделей.
  - Обучения модели и сохранение в файл.
  - Загрузка обученной модели.
  - Классификации категорий тикетов на основе описаний.
  - **Оценки результатов работы модели (в разработке).**

---

## Эндпоинты API

1. **Проверка статуса**
   - **URL:** `/status`
   - **Метод:** `GET`
   - **Описание:** Проверяет доступность API.
   - **Пример запроса:**
     ```bash
     curl http://127.0.0.1:5050/status
     ```

2. **Классификация тикета**
   - **URL:** `/categorize`
   - **Метод:** `POST`
   - **Описание:** Классифицирует тикет на основе его описания.
   - **Пример запроса:**
     ```bash
     curl -X POST http://127.0.0.1:5050/categorize -H "Content-Type: application/json" -d '{"description": "Ticket description example"}'
     ```

3. **Управление данными**
   - **URL:** `/data`
   - **Методы:** `POST`, `GET`
   - **Описание:**
     - `POST`: Загрузка нового CSV-файла для обучения модели.
     - `GET`: Получение текущих данных.
   - **Пример загрузки файла:**
     ```bash
     curl -X POST -F 'file=@path_to_file.csv' http://127.0.0.1:5050/data
     ```

4. **Загрузка модели**
   - **URL:** `/load-model`
   - **Метод:** `POST`
   - **Описание:** Загружает пользовательскую модель и векторизатор.
   - **Пример загрузки модели:**
     ```bash
     curl -X POST -F 'model=@path_to_model.pkl' -F 'vectorizer=@path_to_vectorizer.pkl' http://127.0.0.1:5050/load-model
     ```

---

## Настройка данных

### Подготовка данных
1. Убедитесь, что ваш CSV-файл содержит, как минимум, два обязательных столбца:
   - `Description` — текстовое описание тикета.
   - `Type` — категория тикета.

2. Загрузите данные через `/data` или поместите файл `tickets.csv` в папку `data`.

---

## Требования

- Python 3
- PostgreSQL
- Установленные зависимости из `requirements.txt`