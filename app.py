from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your trained model
model = tf.keras.models.load_model('savingmodel.h5')

# Define class labels
class_indices = {'AppleCedarRust': 0, 'AppleScab': 1, 'Healthy': 2}  # Update with your classes
class_labels = {v: k for k, v in class_indices.items()}

def predict(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return class_labels[predicted_class]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_class = predict(filepath)
        return render_template('result.html', image_url=filepath, prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
