import pytest
from playwright.sync_api import sync_playwright
import time


@pytest.fixture(scope="module")
def start_streamlit_app():
    import subprocess
    import os

    process = subprocess.Popen(["streamlit", "run", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)
    yield
    process.terminate()


def test_status_page(start_streamlit_app):
    """Тест проверки страницы статуса сервера"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False) 
        page = browser.new_page()
        page.goto("http://localhost:8501")

        page.get_by_text("Статус сервера").click()
        time.sleep(1)

        assert page.get_by_text("Сервер работает!").is_visible()
        browser.close()


def test_ticket_classification(start_streamlit_app):
    """Тест страницы классификации тикетов"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        page.get_by_text("Классификация тикетов").click()
        time.sleep(1)

        page.fill("textarea", "Test ticket description")
        page.fill("input[aria-label='Введите ID тикета (опционально):']", "123")
        page.fill("input[aria-label='Введите заголовок тикета (опционально):']", "Test Title")

        page.get_by_text("Классифицировать").click()
        page.get_by_text("Классифицировать").click()
        time.sleep(2)

        assert page.get_by_text("Предсказанная категория").is_visible()
        browser.close()


def test_train_model_page(start_streamlit_app):
    """Тест страницы обучения модели"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        page.get_by_text("Обучение модели").click()
        time.sleep(1)

        page.get_by_role("checkbox", name="Подгружать данные из базы перед обучением").check()

        page.get_by_text("Обучить модель").click()
        time.sleep(5)

        assert page.get_by_text("Обучение завершено").is_visible()
        browser.close()


def test_data_management_page(start_streamlit_app):
    """Тест страницы управления данными"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        page.get_by_text("Управление данными").click()
        time.sleep(1)

        file_input = page.get_by_label("Загрузите CSV файл для обновления данных")
        file_input.set_input_files("test_data.csv")
        page.get_by_text("Посмотреть текущие данные").click()
        time.sleep(2)

        assert page.get_by_text("Test ticket 1").is_visible()
        browser.close()


def test_visualization_page(start_streamlit_app):
    """Тест страницы визуализации"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        page.get_by_text("Визуализация").click()
        time.sleep(1)

        assert page.locator("canvas").is_visible()
        browser.close()


def test_model_management_page(start_streamlit_app):
    """Тест работы с моделью"""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("http://localhost:8501")

        page.get_by_text("Работа с моделью").click()
        time.sleep(1)

        page.get_by_text("Скачать выбранный файл").click()
        time.sleep(2)

        assert page.get_by_role("button", name="Нажмите для скачивания модели").is_visible()

        page.set_input_files("input[name='model_uploader']", "test_model.pkl")
        page.set_input_files("input[name='vectorizer_uploader']", "test_vectorizer.pkl")
        
        page.get_by_text("Загрузить файлы").click()
        page.get_by_text("Загрузить файлы").click()
        time.sleep(2)

        assert page.get_by_text("Файлы успешно загружены и применены.").is_visible()
        browser.close()