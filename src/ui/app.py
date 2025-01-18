import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:5050"

st.set_page_config(page_title="AI Ticket Classifier", layout="wide")

st.title("AI Ticket Classifier")

# ====================== Категоризация тикета ======================
st.header("Категоризация тикета")
with st.form("categorize_form"):
    ticket_id = st.text_input("ID тикета (необязательно):")
    title = st.text_input("Название тикета (необязательно):")
    description = st.text_area("Описание тикета:", height=150)
    submitted = st.form_submit_button("Классифицировать")

    if submitted:
        if not description:
            st.error("Описание тикета обязательно!")
        else:
            payload = {
                "id": ticket_id,
                "title": title,
                "description": description
            }
            response = requests.post(f"{API_URL}/categorize", json=payload)
            if response.status_code == 200:
                st.success(f"Предсказанная категория: {response.json()['predicted_type']}")
            else:
                st.error(f"Ошибка: {response.json().get('error', 'Неизвестная ошибка')}")

# ====================== Обучение модели ======================
st.header("Обучение модели")
with st.form("train_model_form"):
    load_from_db = st.checkbox("Догрузить данные из базы данных")
    train_submitted = st.form_submit_button("Запустить обучение")

    if train_submitted:
        payload = {"load_from_db": load_from_db}
        response = requests.post(f"{API_URL}/train-model", json=payload)
        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(f"Ошибка: {response.json().get('error', 'Неизвестная ошибка')}")

# ====================== Управление данными ======================
st.header("Управление данными")
uploaded_file = st.file_uploader("Загрузить CSV-файл с данными", type=["csv"])
if st.button("Загрузить данные"):
    if uploaded_file is not None:
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_URL}/data", files=files)
        if response.status_code == 200:
            st.success("Файл успешно загружен!")
        else:
            st.error(f"Ошибка загрузки файла: {response.json().get('error', 'Неизвестная ошибка')}")
    else:
        st.error("Файл не выбран!")

if st.button("Получить текущие данные"):
    response = requests.get(f"{API_URL}/data")
    if response.status_code == 200:
        data = response.json()["data"]
        df = pd.DataFrame(data)
        st.write(df)
    else:
        st.error(f"Ошибка: {response.json().get('error', 'Неизвестная ошибка')}")

# ====================== Загрузка модели ======================
st.header("Загрузка пользовательской модели")
model_file = st.file_uploader("Файл модели (PKL):", type=["pkl"], key="model_upload")
vectorizer_file = st.file_uploader("Файл векторизатора (PKL):", type=["pkl"], key="vectorizer_upload")
if st.button("Загрузить модель и векторизатор"):
    if model_file and vectorizer_file:
        files = {
            "model": model_file.getvalue(),
            "vectorizer": vectorizer_file.getvalue()
        }
        response = requests.post(f"{API_URL}/load-model", files=files)
        if response.status_code == 200:
            st.success(response.json()["message"])
        else:
            st.error(f"Ошибка загрузки модели: {response.json().get('error', 'Неизвестная ошибка')}")
    else:
        st.error("Оба файла (модель и векторизатор) должны быть предоставлены!")

# ====================== Экспорт данных ======================
st.header("Экспорт данных")
if st.button("Экспортировать данные из базы в tickets.csv"):
    response = requests.post(f"{API_URL}/export-tickets")
    if response.status_code == 200:
        st.success("Данные успешно экспортированы в tickets.csv")
    else:
        st.error(f"Ошибка: {response.json().get('error', 'Неизвестная ошибка')}")