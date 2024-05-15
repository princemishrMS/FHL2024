import os
import time
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify, send_file
import shutil

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
app = Flask(__name__)
SAVED_MODEL_PATH = "archive"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'saved_images'

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def enhance_image(image_path):
    image = Image.open(image_path)
    
    # Example of enhancing the image quality
    enhanced_image = image.filter(ImageFilter.SHARPEN)
    
    # You can apply other enhancement techniques here
    
    # Save the enhanced image
    enhanced_image_path = os.path.join(UPLOAD_FOLDER, "enhanced_" + os.path.basename(image_path))
    enhanced_image.save(enhanced_image_path)
    
    return enhanced_image_path

def save_image(image, filename):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(filename)

def clear_upload_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

@app.after_request
def clear_upload_folder_after_request(response):
    clear_upload_folder(UPLOAD_FOLDER)
    return response

@app.route('/', methods=['GET'])
def home():
    return "Hello, World!"

@app.route('/super_resolution', methods=['POST'])
def super_resolution():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Save uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Preprocess image
        hr_image = preprocess_image(image_path)

        # Load model
        try:
            model = hub.load(SAVED_MODEL_PATH)
        except Exception as e:
            return jsonify({"error": f"Error loading model: {e}"}), 500

        # Generate super-resolution image
        try:
            fake_image = model(hr_image)
            fake_image = tf.squeeze(fake_image)
        except Exception as e:
            return jsonify({"error": f"Error during model inference: {e}"}), 500

        # Generate filename for super-resolution image
        filename = file.filename.split('.')[0] + "_super_resolution.jpg"
        output_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save super-resolution image
        try:
            save_image(fake_image, output_path)
        except Exception as e:
            return jsonify({"error": f"Error saving image: {e}"}), 500

        # Return the saved image file
        return send_file(output_path, as_attachment=True), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

@app.route('/enhance_quality', methods=['POST'])
def enhance_quality():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Save uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Enhance the quality of the image
        try:
            enhanced_image_path = enhance_image(image_path)
        except Exception as e:
            return jsonify({"error": f"Error enhancing image: {e}"}), 500

        # Return the enhanced image file
        return send_file(enhanced_image_path, as_attachment=True), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)
