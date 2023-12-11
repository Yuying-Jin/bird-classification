# Import modules
import os
from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

# Create a Flask web application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for handling requests from different origins
CORS(app)


# Define a route for handling the result page
@app.route('/result', methods=["GET", "POST"])
def result():
    try:
        # Check if 'imageInput' is present in the uploaded files
        if 'imageInput' not in request.files:
            return 'No imageInput or empty file'
        # Get the uploaded image file
        imageInput = request.files['imageInput']

        # Save the uploaded image to the specified path
        saved_path = os.path.join("upload_image", imageInput.filename)
        imageInput.save(saved_path)

        # Use YOLO model for prediction with a confidence threshold of 0.3
        result = model.predict(saved_path, conf=0.3, save_conf=True)

        # Get the predicted label and confidence level
        predicted_label = result[0].names.get(result[0].probs.top1)
        confidence = result[0].probs.top1conf.item()

        # Render the result template and present predicted label and confidence
        return render_template('result.html', predicted_label=predicted_label.lower(), confidence=round(confidence, 2))

    except Exception as e:
        # Handle any exceptions that may occur during the process
        return f"Error: {str(e)}"


# Define routes for the home page
@app.route('/', methods=["GET"])
@app.route('/home.html', methods=["GET"])
def get_index():
    # Serve the home.html file from the root directory
    return send_from_directory('', 'home.html', mimetype='text/html')


# Entry point of the application
if __name__ == '__main__':
    # Load the YOLO model
    model = YOLO('best.pt')
    # Run the Flask application on port 8000
    app.run(port=8000)
