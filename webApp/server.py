import os

from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)


@app.route('/result', methods=["GET", "POST"])
def result():
    try:
        if 'imageInput' not in request.files:
            return 'No imageInput or empty file'
        imageInput = request.files['imageInput']

        saved_path = os.path.join("upload_image", imageInput.filename)
        imageInput.save(saved_path)

        result = model.predict(saved_path, conf=0.5, save_conf=True)

        predicted_label = result[0].names.get(result[0].probs.top1)
        confidence = result[0].probs.top1conf.item()

        return render_template('result.html', predicted_label=predicted_label, confidence=confidence)

    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/', methods=["GET"])
@app.route('/home.html', methods=["GET"])
def get_index():
    return send_from_directory('', 'home.html', mimetype='text/html')


if __name__ == '__main__':
    model = YOLO('best.pt')
    app.run(port=8000)
