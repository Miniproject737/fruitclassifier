import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename  # Ensures safe file saving

app = Flask(__name__)

# Load the trained model (.h5 format)
MODEL_PATH = "fruit_classifier_model.h5"  # Ensure this matches your file name
model = tf.keras.models.load_model(MODEL_PATH)

# Debug: Print model input shape
print("Model Input Shape:", model.input_shape)  # Should match (None, 224, 224, 3)

# Allowed image extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    """Check if the file extension is valid."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Load and preprocess image for model prediction."""
    img = Image.open(image_path).convert("RGB")  # Convert to RGB (3 channels)
    img = img.resize((224, 224))  # Resize to model's expected input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/")
def index():
    """Render home page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and prediction."""
    if "file" not in request.files:
        return render_template("index.html", message="No file uploaded")

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return render_template("index.html", message="Invalid file type. Upload a JPG or PNG.")

    # Secure filename and save the uploaded image
    filename = secure_filename(file.filename)
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)  # Create folder if not exists
    image_path = os.path.join(upload_folder, filename)
    file.save(image_path)

    # Preprocess image and predict
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)  # Get class index
    confidence = float(np.max(prediction) * 100)  # Convert to percentage

    # Define fruit classes (ensure these match your trained model's classes)
    fruit_classes = ["Apple", "banana", "guava", "mango","orange", "papaya", "pineapple", "waterapple", "watermelon"]
    predicted_label = fruit_classes[predicted_class] if predicted_class < len(fruit_classes) else "Unknown"

    return render_template("result.html", fruit=predicted_label, confidence=confidence, image=f"uploads/{filename}")

if __name__ == "__main__":
     port = int(os.environ.get("PORT", 5000))  # Railway provides a PORT variable
    app.run(host="0.0.0.0", port=port)
