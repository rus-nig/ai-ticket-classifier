import streamlit as st
import requests
import pandas as pd
from io import StringIO

API_URL = "http://127.0.0.1:5050"


st.sidebar.title("Навигация")
menu = st.sidebar.radio(
    "Выберите действие",
    ["Статус сервера", "Классификация тикетов", "Обучение модели", "Управление данными", "Визуализация"]
)

# Вкладка: Статус сервера
if menu == "Статус сервера":
    st.title("Проверка статуса сервера")
    try:
        response = requests.get(f"{API_URL}/status")
        if response.status_code == 200:
            st.success(f"Сервер работает! Ответ: {response.json()}")
        else:
            st.error(f"Ошибка соединения с сервером. Код ответа: {response.status_code}")
    except Exception as e:
        st.error(f"Ошибка: {e}")

# Вкладка: Классификация тикетов
elif menu == "Классификация тикетов":
    st.title("Классификация тикетов")
    description = st.text_area("Введите описание тикета:")
    ticket_id = st.text_input("Введите ID тикета (опционально):")
    title = st.text_input("Введите заголовок тикета (опционально):")

    if st.button("Классифицировать"):
        payload = {"description": description, "id": ticket_id, "title": title}
        try:
            response = requests.post(f"{API_URL}/categorize", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Предсказанная категория: {result['predicted_type']}")
                st.write(result)
            else:
                st.error(f"Ошибка: {response.json()['error']}")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# Вкладка: Обучение модели
elif menu == "Обучение модели":
    st.title("Обучение модели")
    load_from_db = st.checkbox("Подгружать данные из базы перед обучением")

    if st.button("Обучить модель"):
        payload = {"load_from_db": load_from_db}
        try:
            response = requests.post(f"{API_URL}/train-model", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])
                st.write(result)
            else:
                st.error(f"Ошибка: {response.json()['error']}")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# Вкладка: Управление данными
elif menu == "Управление данными":
    st.title("Управление данными")
    uploaded_file = st.file_uploader("Загрузите CSV файл для обновления данных", type=["csv"])

    if uploaded_file is not None:
        try:
            response = requests.post(
                f"{API_URL}/data",
                files={"file": ("tickets.csv", uploaded_file.getvalue())}
            )
            if response.status_code == 200:
                st.success("Данные успешно загружены!")
            else:
                st.error(f"Ошибка загрузки данных: {response.json()['error']}")
        except Exception as e:
            st.error(f"Ошибка: {e}")

    if st.button("Посмотреть текущие данные"):
        try:
            response = requests.get(f"{API_URL}/data")
            if response.status_code == 200:
                data = response.json()["data"]
                st.write(pd.DataFrame(data))
            else:
                st.error(f"Ошибка загрузки данных: {response.json()['error']}")
        except Exception as e:
            st.error(f"Ошибка: {e}")

# Вкладка: Визуализация
elif menu == "Визуализация":
    st.title("Визуализация данных")
    try:
        response = requests.get(f"{API_URL}/data")
        if response.status_code == 200:
            data = pd.DataFrame(response.json()["data"])
            st.write("Распределение категорий тикетов:")
            st.bar_chart(data["Type"].value_counts())
        else:
            st.error(f"Ошибка загрузки данных: {response.json()['error']}")
    except Exception as e:
        st.error(f"Ошибка: {e}")