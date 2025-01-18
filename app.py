# app.py
from flask import Flask, render_template, request, jsonify
from translate_web import translate
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        text = request.form.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        translation = translate(text)
        return jsonify({'translation': translation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
