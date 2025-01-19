import os
import sys
import multiprocessing

def run_service():
    os.system("python -m src.routes.service")

def run_app():
    os.system("streamlit run src/ui/app.py")

if __name__ == "__main__":
    if len(sys.argv) == 1:  # Если параметры не переданы, запускаем оба компонента
        print("Параметры не указаны. Запуск Flask API и Streamlit GUI...")
        service_process = multiprocessing.Process(target=run_service)
        app_process = multiprocessing.Process(target=run_app)

        service_process.start()
        app_process.start()

        service_process.join()
        app_process.join()

    elif len(sys.argv) == 2:
        command = sys.argv[1]
        if command == "service":
            print("Запуск Flask API...")
            run_service()
        elif command == "app":
            print("Запуск Streamlit GUI...")
            run_app()
        else:
            print("Неверная команда. Используйте 'service' для запуска API, 'app' для запуска интерфейса, или запустите без параметров для запуска обоих.")
            sys.exit(1)
    else:
        print("Неверное использование. Укажите только один параметр или не указывайте вовсе.")
        sys.exit(1)