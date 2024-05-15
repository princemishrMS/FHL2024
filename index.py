import os
import time
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify
import shutil
import io
from OpenSSL import SSL

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
app = Flask(__name__)
SAVED_MODEL_PATH = "archive"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define SSL certificate and key file paths 
CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"

# Create SSL context
context = SSL.Context(SSL.TLSv1_2_METHOD)
context.use_certificate_file(CERT_FILE)
context.use_privatekey_file(KEY_FILE)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    hr_image = tf.image.decode_image(image.read(), channels=3)
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32) / 255.0
    return tf.expand_dims(hr_image, 0)

def enhance_image(image):
    image = Image.open(io.BytesIO(image.read()))
    
    # Example of enhancing the image quality
    enhanced_image = image.filter(ImageFilter.SHARPEN)
    
    # You can apply other enhancement techniques here
    
    # Convert the enhanced image to bytes
    enhanced_image_bytes = io.BytesIO()
    enhanced_image.save(enhanced_image_bytes, format='JPEG')
    enhanced_image_bytes.seek(0)
    
    return enhanced_image_bytes.getvalue()

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
        try:
            # Preprocess image
            hr_image = preprocess_image(file)

            # Load model
            model = hub.load(SAVED_MODEL_PATH)

            # Generate super-resolution image
            fake_image = model(hr_image)
            fake_image = tf.squeeze(fake_image)

            # Convert the super-resolution image to bytes
            output_image_bytes = io.BytesIO()
            Image.fromarray(np.uint8(fake_image.numpy() * 255)).save(output_image_bytes, format='JPEG')
            output_image_bytes.seek(0)

            # Return the super-resolution image
            return output_image_bytes.getvalue(), 200
        except Exception as e:
            return jsonify({"error": f"Error: {e}"}), 500
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
        try:
            # Enhance the quality of the image
            enhanced_image_bytes = enhance_image(file)

            # Return the enhanced image
            return enhanced_image_bytes, 200
        except Exception as e:
            return jsonify({"error": f"Error: {e}"}), 500
    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=443,debug=True,ssl_context=context)

