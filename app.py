"""
Flask Web Application for CIFAR-10 Image Classification
Deploys a deep learning model to classify images into 10 categories.
"""

import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt

# Load TensorFlow/Keras for model inference
from tensorflow.keras.models import load_model

# Initialize Flask application
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SECRET_KEY'] = 'cifar10-classification-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# CIFAR-10 class labels (in order of model output indices)
CLASS_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Load the trained model once at startup - use absolute path based on script location
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'cifar10_model.h5')
model = None


def load_cifar10_model():
    """Load the pre-trained CIFAR-10 CNN model from model/cifar10_model.h5."""
    global model
    # Try primary path (script directory), then fallback to cwd
    paths_to_try = [
        MODEL_PATH,
        os.path.join(os.getcwd(), 'model', 'cifar10_model.h5'),
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            model = load_model(path)
            return True
    return False


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    """
    Predict the class of an image using the CIFAR-10 model.
    
    Steps:
    1. Load image using PIL
    2. Resize image to (32, 32) - CIFAR-10 input size
    3. Convert to numpy array
    4. Normalize pixel values (divide by 255)
    5. Expand dimensions for batch prediction
    6. Use model.predict()
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (predicted_label, probability_scores)
    """
    # Load image using PIL
    img = Image.open(image_path)
    
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image to (32, 32) - CIFAR-10 standard input size
    img = img.resize((32, 32), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize pixel values (0-255 -> 0-1) - matches training preprocessing
    img_array = img_array.astype(np.float32) / 255.0
    
    # Expand dimensions to add batch dimension: (32, 32, 3) -> (1, 32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Model prediction
    predictions = model.predict(img_array, verbose=0)
    probability_scores = predictions[0]
    
    # Get predicted class index and label
    predicted_idx = np.argmax(probability_scores)
    predicted_label = CLASS_LABELS[predicted_idx]
    
    return predicted_label, probability_scores


def create_probability_chart(probability_scores, output_path):
    """
    Generate a bar chart showing probability distribution for all 10 classes.
    Saves the chart as an image for display on the result page.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(CLASS_LABELS)))
    bars = ax.bar(CLASS_LABELS, probability_scores * 100, color=colors)
    
    # Highlight the predicted class
    pred_idx = np.argmax(probability_scores)
    bars[pred_idx].set_color('#e74c3c')
    bars[pred_idx].set_edgecolor('black')
    bars[pred_idx].set_linewidth(2)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title('Class Probability Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/')
def index():
    """Render the home page with image upload interface."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if model is None:
        flash('Model not loaded. Please place cifar10_model.h5 in the model folder.', 'error')
        return redirect(url_for('index'))
    
    # Check if file was uploaded
    if 'image' not in request.files:
        flash('No image file uploaded. Please select an image.', 'error')
        return redirect(url_for('index'))
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No image selected. Please choose an image to upload.', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded image
        filename = secure_filename(file.filename)
        # Add timestamp to avoid overwriting
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(image_path)
        
        # Run prediction
        predicted_label, probability_scores = predict_image(image_path)
        
        # Get confidence (max probability)
        confidence = float(np.max(probability_scores)) * 100
        
        # Generate probability bar chart
        chart_filename = f"chart_{timestamp}.png"
        chart_path = os.path.join('static', chart_filename)
        create_probability_chart(probability_scores, chart_path)
        
        # Prepare probability dict for template
        prob_dict = {label: float(prob) * 100 for label, prob in zip(CLASS_LABELS, probability_scores)}
        
        return render_template(
            'result.html',
            image_filename=unique_filename,
            predicted_class=predicted_label,
            confidence=round(confidence, 2),
            chart_path=chart_filename,
            prob_dict=prob_dict
        )
        
    except Exception as e:
        flash(f'Prediction error: {str(e)}', 'error')
        return redirect(url_for('index'))


# Load model when application starts
if __name__ == '__main__':
    print(f"Model path: {MODEL_PATH}", flush=True)
    print(f"Model exists: {os.path.exists(MODEL_PATH)}", flush=True)
    if load_cifar10_model():
        print("CIFAR-10 model loaded successfully!", flush=True)
    else:
        print("WARNING: cifar10_model.h5 not found. Run train_model.py first to train the model.", flush=True)
    
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
