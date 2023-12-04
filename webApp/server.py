from flask import Flask, send_from_directory # Import Flask module
from flask_cors import CORS
import json
from ultralytics import YOLO
app = Flask(__name__)
CORS(app)



@app.route('/result', methods=["GET"])
def result():
    #result = model.predict(imageInput)
    result = "Prediction"
    resultDict = { "Result": result}
    resultString = json.dumps(resultDict)
    return resultString


@app.route('/', methods=["GET"])
@app.route('/home.html', methods=["GET"])
def get_index():
    return send_from_directory('', 'home.html', mimetype='text/html')


@app.route('/home.js', methods=["GET"])
def get_main():
    return send_from_directory('', 'home.js', mimetype='text/javascript')


if __name__ == '__main__':
    model = YOLO('best.pt')
    app.run(port = 8000)
