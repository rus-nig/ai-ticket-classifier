from flask import Flask, jsonify
import pandas as pd


app = Flask(__name__)


# ==================== Эндпоинты ====================

@app.route('/ping', methods=['GET'])
def status():
    """Проверка работы API."""
    return jsonify({
        "status": "API работает"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)