# Bird Classification

## Overview
This project focuses on bird classification using two machine learning models: YOLOv8 and ResNet50.  The web application uses the YOLOv8 model to classify birds. It is built with the Flask framework in Python, providing a user-friendly interface to upload and classify bird images.

## Prerequisites
- Python `3.8` or higher installed

## Setup

1. Clone the repository or Download the codebase:

    ```bash
    git clone https://github.com/Yuying-Jin/bird-classification.git # clone the repo
    ```

2. Navigate to the app directory:

    ```bash
    cd bird-classification/webApp
    ```

3. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
## Running the Application

1. Start the Flask application:

    ```bash
    python server.py
    ```

2. Open a web browser and navigate to `http://localhost:8000`.

3. Upload a bird image using the provided interface.

## Usage

- The web app will process the uploaded image using the YOLOv8 model and display the results, including the top1 class label and its confidence in two decimals. 
