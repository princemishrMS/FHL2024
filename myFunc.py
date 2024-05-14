import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify

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

def preprocess_image(image):
    """Preprocesses the image to make it model-ready"""
    hr_image = image.convert("RGB")
    hr_size = (np.array(hr_image.size) // 4) * 4
    hr_image = hr_image.crop((0, 0, hr_size[0], hr_size[1]))
    hr_image = np.array(hr_image) / 255.0
    hr_image = tf.convert_to_tensor(hr_image, dtype=tf.float32)  # Convert to float32
    hr_image = tf.expand_dims(hr_image, 0)
    return hr_image

def save_image(image, filename):
    """
    Saves unscaled Tensor Images.
    Args:
        image: 3D image tensor. [height, width, channels]
        filename: Name of the file to save.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save("%s.jpg" % filename)
    print("Saved as %s.jpg" % filename)

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
        image = Image.open(file)
        hr_image = preprocess_image(image)

        try:
            model = hub.load(SAVED_MODEL_PATH)
            # Inspect the model's signatures to help debug
            signatures = list(model.signatures.keys())
            print(f"Available signatures: {signatures}")
            infer = model.signatures['serving_default']
            print(f"Inference signature: {infer.structured_input_signature}")
        except Exception as e:
            return jsonify({"error": f"Error loading model: {e}"}), 500

        try:
            start = time.time()
            fake_image = infer(hr_image)['output_0']  # Adjust the output key as necessary
            print("DEBUG ========================", fake_image)
            fake_image = tf.squeeze(fake_image)
            print("DEBUG ======================== Time Taken: %f" % (time.time() - start))
        except Exception as e:
            return jsonify({"error": f"Error during model inference: {e}"}), 500

        # Generate a unique filename for the high-resolution image
        filename = str(int(time.time()))
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        # Save the high-resolution image
        save_image(fake_image, file_path)

        # Return the URL of the saved image
        image_url = f"http://your_server_domain/{filename}.jpg"
        return jsonify({"result": "success", "image_url": image_url}), 200
    else:
        return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True)