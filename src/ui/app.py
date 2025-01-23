import streamlit as st
import requests
import pandas as pd
from io import StringIO

API_URL = "http://127.0.0.1:5050"

st.sidebar.title("Навигация")
menu = st.sidebar.radio(
    "Выберите действие",
    ["Статус сервера", "Классификация тикетов", "Обучение модели", "Управление данными", "Визуализация", "Работа с БД", "Работа с моделью"]
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
            
            if data.empty:
                st.warning("Данные отсутствуют для визуализации.")
            else:
                st.write("Выберите тип визуализации данных:")
                visualization_type = st.selectbox(
                    "Тип визуализации",
                    ["Распределение категорий", "Частотный анализ слов", "Динамика по времени"]
                )
                
                if visualization_type == "Распределение категорий":
                    st.subheader("Распределение категорий тикетов")
                    st.bar_chart(data["Type"].value_counts())

                elif visualization_type == "Частотный анализ слов":
                    st.subheader("Частотный анализ слов в описаниях тикетов")
                    all_descriptions = " ".join(data["Description"].dropna().tolist())
                    word_counts = pd.Series(all_descriptions.split()).value_counts().head(20)
                    
                    st.write("Топ-20 самых частых слов:")
                    st.bar_chart(word_counts)

                elif visualization_type == "Динамика по времени":
                    st.subheader("Динамика создания тикетов по времени")
                    
                    if "Created_At" not in data.columns:
                        st.warning("Временная информация отсутствует. Генерируются случайные даты.")
                        import random
                        data["Created_At"] = pd.date_range("2025-01-01", periods=len(data), freq="D")
                    
                    data["Created_At"] = pd.to_datetime(data["Created_At"])
                    tickets_by_date = data.groupby(data["Created_At"].dt.date).size()
                    
                    st.line_chart(tickets_by_date)
                    
                st.subheader("Фильтрация данных")
                category_filter = st.multiselect("Фильтруйте по категориям:", data["Type"].unique())
                
                if category_filter:
                    filtered_data = data[data["Type"].isin(category_filter)]
                    st.write("Отфильтрованные данные:")
                    st.dataframe(filtered_data)
                else:
                    st.info("Выберите категории для фильтрации данных.")
        else:
            st.error(f"Ошибка загрузки данных: {response.json()['error']}")
    except Exception as e:
        st.error(f"Ошибка: {e}")

# Работа с БД
elif menu == "Работа с БД":
    st.title("Работа с БД")
    action = st.selectbox("Выберите действие", ["Получить все записи", "Удалить запись", "Редактировать запись"])

    if action == "Получить все записи":
        if st.button("Получить"):
            response = requests.get(f"{API_URL}/tickets")
            if response.status_code == 200:
                record = response.json().get("tickets", {})
                st.write(record)
            else:
                st.error(f"Ошибка: {response.json().get('error', 'Неизвестная ошибка')}")

    elif action == "Удалить запись":
        record_id = st.text_input("Введите ID записи")
        if st.button("Удалить"):
            response = requests.delete(f"{API_URL}/tickets/{record_id}")
            if response.status_code == 200:
                st.success("Запись успешно удалена")
            else:
                st.error(f"Ошибка: {response.json().get('error', 'Неизвестная ошибка')}")

    elif action == "Редактировать запись":
        record_id = st.text_input("Введите ID записи")
        new_title = st.text_input("Введите новый заголовок (опционально)")
        new_description = st.text_area("Введите новое описание (опционально)")
        new_predicted_type = st.text_input("Введите новую категорию (опционально)")

        if st.button("Редактировать"):
            payload = {
                "title": new_title if new_title else None,
                "description": new_description if new_description else None,
                "predicted_type": new_predicted_type if new_predicted_type else None,
            }
            response = requests.put(f"{API_URL}/tickets/{record_id}", json=payload)
            if response.status_code == 200:
                st.success("Запись успешно обновлена")
            else:
                st.error(f"Ошибка: {response.json().get('error', 'Неизвестная ошибка')}")

# Работа с моделью и векторизатором
elif menu == "Работа с моделью":
    st.title("Работа с моделью и векторизатором")

    st.subheader("Скачивание файлов")
    file_type_download = st.selectbox(
        "Выберите файл для скачивания",
        ["Модель", "Векторизатор"]
    )

    if st.button("Скачать выбранный файл"):
        try:
            type_param = "model" if file_type_download == "Модель" else "vectorizer"
            response = requests.get(f"{API_URL}/model-files?type={type_param}", stream=True)

            if response.status_code == 200:
                st.success(f"Файл {file_type_download.lower()} готов к скачиванию.")
                file_name = f"downloaded_{type_param}.pkl"
                with open(file_name, "wb") as file:
                    file.write(response.content)

                with open(file_name, "rb") as file:
                    st.download_button(
                        label=f"Нажмите для скачивания {file_type_download.lower()}",
                        data=file,
                        file_name=f"{type_param}.pkl",
                        mime="application/octet-stream",
                    )
            else:
                st.error(f"Ошибка при скачивании {file_type_download.lower()}: {response.json().get('error', 'Неизвестная ошибка')}")
        except Exception as e:
            st.error(f"Ошибка: {e}")

    st.subheader("Загрузка файлов")
    uploaded_model = st.file_uploader("Загрузите файл модели", type=["pkl"], key="model_uploader")
    uploaded_vectorizer = st.file_uploader("Загрузите файл векторизатора", type=["pkl"], key="vectorizer_uploader")

    if st.button("Загрузить файлы"):
        if not uploaded_model or not uploaded_vectorizer:
            st.error("Оба файла (модель и векторизатор) должны быть предоставлены.")
        else:
            try:
                response = requests.post(
                    f"{API_URL}/model-files",
                    files={
                        "model": ("model.pkl", uploaded_model.getvalue()),
                        "vectorizer": ("vectorizer.pkl", uploaded_vectorizer.getvalue())
                    }
                )
                if response.status_code == 200:
                    st.success("Файлы успешно загружены и применены.")
                else:
                    st.error(f"Ошибка при загрузке файлов: {response.json().get('error', 'Неизвестная ошибка')}")
            except Exception as e:
                st.error(f"Ошибка: {e}")

st.sidebar.markdown(
    """
    <hr>
    <div style="text-align: center;">
        <a href="http://127.0.0.1:5050/apidocs/" target="_blank" style="text-decoration: none;">
            <button style="
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 15px;
                font-size: 14px;
                cursor: pointer;
                border-radius: 5px;">
                Swagger UI
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)