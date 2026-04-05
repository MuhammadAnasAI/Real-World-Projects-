import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from fastai.vision.all import load_learner, PILImage
import os
import pathlib
from pathlib import Path
import pickle

class WindowsPathUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.WindowsPath
        if module == "pathlib" and name == "PurePosixPath":
            return pathlib.PureWindowsPath
        return super().find_class(module, name)

class PickleModuleWrapper:
    Unpickler = WindowsPathUnpickler
    load = pickle.load
    loads = pickle.loads
    dump = pickle.dump
    dumps = pickle.dumps

# ------------------------------------------------------------
# 1. Load the fastai learner (cached for performance)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the fastai learner from the pickle file."""
    try:
        learn = load_learner("ai_vs_real_model.pkl", cpu=True, pickle_module=PickleModuleWrapper)
        return learn
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# ------------------------------------------------------------
# 2. Helper function to preprocess the uploaded image
# ------------------------------------------------------------
def preprocess_image(image):
    """
    Convert PIL image to fastai PILImage for prediction.
    Convert to RGB and ensure proper format for the model.
    """
    try:
        # Convert to RGB if image has alpha channel or is in different mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return PILImage.create(image)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# ------------------------------------------------------------
# 3. Prediction function
# ------------------------------------------------------------
def predict(learn, img):
    """
    Run prediction using fastai learner.
    Returns human-readable label, prediction index, confidence, and probability array.
    """
    try:
        pred, pred_idx, probs = learn.predict(img)
        pred_idx = int(pred_idx)
        confidence = float(probs[pred_idx])

        # Map numeric labels to friendly names when necessary.
        if isinstance(pred, (np.integer, int)):
            pred_label = "AI Generated" if pred_idx == 1 else "Real"
        else:
            pred_label = str(pred)

        return pred_label, pred_idx, confidence, probs.numpy()
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None, None

# ------------------------------------------------------------
# 4. Plot a bar chart of prediction probabilities
# ------------------------------------------------------------
def plot_probabilities(probs, class_names):
    fig, ax = plt.subplots(figsize=(8, 5))
    formatted_names = [str(name) for name in class_names]
    colors = ['green' if str(name).lower() == 'real' else 'red' for name in formatted_names]
    bars = ax.bar(formatted_names, probs, color=colors, alpha=0.7)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, v in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

# ------------------------------------------------------------
# 5. Dataset visualization functions
# ------------------------------------------------------------
def get_dataset_stats():
    """Get statistics about the dataset."""
    base_path = Path(".")
    ai_path = base_path / "Ai_generated_dataset"
    real_path = base_path / "real_dataset"
    
    stats = {}
    categories = ['animals', 'city', 'food', 'nature', 'people']
    
    for category in categories:
        ai_count = len(list((ai_path / category).glob("*.jpg"))) if (ai_path / category).exists() else 0
        real_count = len(list((real_path / category).glob("*.jpg"))) if (real_path / category).exists() else 0
        stats[category] = {'ai': ai_count, 'real': real_count, 'total': ai_count + real_count}
    
    return stats

def plot_dataset_distribution():
    """Plot the distribution of images in the dataset."""
    stats = get_dataset_stats()
    
    categories = list(stats.keys())
    ai_counts = [stats[cat]['ai'] for cat in categories]
    real_counts = [stats[cat]['real'] for cat in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.35
    
    ax.bar(x - width/2, ai_counts, width, label='AI Generated', color='red', alpha=0.7)
    ax.bar(x + width/2, real_counts, width, label='Real', color='green', alpha=0.7)
    
    ax.set_xlabel('Categories', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Dataset Distribution by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

# ------------------------------------------------------------
# 6. Model accuracy testing function
# ------------------------------------------------------------
def test_model_accuracy():
    """Test model accuracy on sample images from the dataset."""
    try:
        base_path = Path(".")
        ai_path = base_path / "Ai_generated_dataset"
        real_path = base_path / "real_dataset"
        
        correct_predictions = 0
        total_predictions = 0
        results = []
        
        # Test on real images
        if real_path.exists():
            for category in ['animals', 'city', 'food', 'nature', 'people']:
                cat_path = real_path / category
                if cat_path.exists():
                    files = list(cat_path.glob("*.jpg")) + list(cat_path.glob("*.jpeg")) + list(cat_path.glob("*.png"))
                    if files:
                        # Test first 2 images per category
                        for img_file in files[:2]:
                            try:
                                img = Image.open(img_file).convert('RGB')
                                pil_img = PILImage.create(img)
                                pred, pred_idx, probs = model.predict(pil_img)
                                
                                # Expected: Real (index 0)
                                expected_idx = 0
                                is_correct = int(pred_idx) == expected_idx
                                correct_predictions += int(is_correct)
                                total_predictions += 1
                                
                                results.append({
                                    'file': img_file.name,
                                    'category': category,
                                    'expected': 'Real',
                                    'predicted': 'AI Generated' if pred_idx == 1 else 'Real',
                                    'confidence': float(probs[pred_idx]),
                                    'correct': is_correct
                                })
                            except Exception as e:
                                print(f"Warning: Error testing {img_file.name}: {e}")
        
        # Test on AI images
        if ai_path.exists():
            for category in ['animals', 'city', 'food', 'nature', 'people']:
                cat_path = ai_path / category
                if cat_path.exists():
                    files = list(cat_path.glob("*.jpg")) + list(cat_path.glob("*.jpeg")) + list(cat_path.glob("*.png"))
                    if files:
                        # Test first 2 images per category
                        for img_file in files[:2]:
                            try:
                                img = Image.open(img_file).convert('RGB')
                                pil_img = PILImage.create(img)
                                pred, pred_idx, probs = model.predict(pil_img)
                                
                                # Expected: AI Generated (index 1)
                                expected_idx = 1
                                is_correct = int(pred_idx) == expected_idx
                                correct_predictions += int(is_correct)
                                total_predictions += 1
                                
                                results.append({
                                    'file': img_file.name,
                                    'category': category,
                                    'expected': 'AI Generated',
                                    'predicted': 'AI Generated' if pred_idx == 1 else 'Real',
                                    'confidence': float(probs[pred_idx]),
                                    'correct': is_correct
                                })
                            except Exception as e:
                                print(f"Warning: Error testing {img_file.name}: {e}")
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return accuracy, results
        
    except Exception as e:
        print(f"Error: Error testing model accuracy: {e}")
        return 0, []

# ------------------------------------------------------------
# 5. Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI vs Real Image Detector", 
    layout="wide",
    page_icon="🖼️",
    initial_sidebar_state="expanded"
)

st.title("🖼️ AI vs Real Image Detector")
st.markdown("**Professional AI-powered image classification tool** - Upload an image to determine if it's AI-generated or real.")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Image Classification", "📊 Dataset Analysis", "🎯 Model Accuracy", "ℹ️ Model Information"])

# Sidebar for model info and accuracy
with st.sidebar:
    st.header("📊 Model Information")
    st.info("**Model**: ResNet34 fine-tuned with FastAI\n**File**: `ai_vs_real_model.pkl`")
    
    # Model accuracy from validation
    st.subheader("🎯 Model Performance")
    accuracy = 0.9146  # From notebook evaluation
    st.metric("Validation Accuracy", f"{accuracy:.2%}")
    
    # Additional metrics
    st.markdown("**Additional Metrics:**")
    st.markdown("- **Precision**: 0.92")
    st.markdown("- **Recall**: 0.91") 
    st.markdown("- **F1-Score**: 0.91")
    
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Upload an image (JPG/PNG)")
    st.markdown("2. AI model analyzes patterns")
    st.markdown("3. Get instant classification")
    st.markdown("4. View confidence scores")

# Tab 1: Image Classification
with tab1:
    st.markdown("### 📤 Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], help="Upload an image to classify as AI-generated or real")

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🖼️ Uploaded Image")
            st.image(image, caption="Your uploaded image", use_container_width=True)
        
        # Preprocess and predict
        img = preprocess_image(image)
        if img is not None:
            with st.spinner("🔍 Analyzing image..."):
                pred_label, pred_idx, confidence, probs = predict(model, img)
            
            if pred_label is not None:
                with col2:
                    st.markdown("### 🎯 Prediction Result")
                    
                    # Display result with appropriate styling
                    label_text = str(pred_label).lower()
                    if "ai" in label_text:
                        st.error(f"🤖 **AI-Generated Image**")
                        st.markdown(f"**Confidence**: {confidence:.2%}")
                    else:
                        st.success(f"📸 **Real Image**")
                        st.markdown(f"**Confidence**: {confidence:.2%}")
                    
                    # Progress bar for confidence
                    st.progress(confidence)
                    
                    # Plot probability bar chart
                    st.markdown("### 📊 Confidence Scores")
                    class_names = ["Real", "AI Generated"]
                    fig = plot_probabilities(probs, class_names)
                    st.pyplot(fig)
                    
                    # Optional: show raw probabilities
                    with st.expander("🔍 Detailed Probabilities"):
                        for i, (name, prob) in enumerate(zip(class_names, probs)):
                            st.write(f"**{name}**: {prob:.4f}")
        
        else:
            st.error("❌ Image preprocessing failed. Please try another image.")

# Tab 2: Dataset Analysis
with tab2:
    st.markdown("### 📊 Dataset Overview")
    st.markdown("Explore the distribution and statistics of the training dataset.")
    
    # Dataset statistics
    stats = get_dataset_stats()
    total_images = sum(stat['total'] for stat in stats.values())
    total_ai = sum(stat['ai'] for stat in stats.values())
    total_real = sum(stat['real'] for stat in stats.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("AI Generated", total_ai)
    with col3:
        st.metric("Real Images", total_real)
    
    # Dataset distribution plot
    st.markdown("### 📈 Dataset Distribution")
    fig = plot_dataset_distribution()
    st.pyplot(fig)
    
    # Category breakdown
    st.markdown("### 📋 Category Breakdown")
    for category, counts in stats.items():
        st.markdown(f"**{category.title()}**: {counts['ai']} AI + {counts['real']} Real = {counts['total']} total")

# Tab 3: Model Accuracy Testing
with tab3:
    st.markdown("### 🎯 Model Accuracy Testing")
    st.markdown("Test the model's performance on sample images from the training dataset.")
    
    if st.button("🧪 Run Accuracy Test", help="Test model accuracy on dataset samples"):
        with st.spinner("Testing model accuracy..."):
            accuracy, results = test_model_accuracy()
            
            if results:
                st.success(f"✅ Accuracy Test Complete! Model accuracy: **{accuracy:.2%}**")
                
                # Display results in a table
                st.markdown("### 📊 Test Results")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tests", len(results))
                with col2:
                    correct = sum(1 for r in results if r['correct'])
                    st.metric("Correct Predictions", correct)
                with col3:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                
                # Confidence distribution
                st.markdown("### 📈 Confidence Distribution")
                confidences = [r['confidence'] for r in results]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Prediction Confidence Distribution')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Error analysis
                errors = [r for r in results if not r['correct']]
                if errors:
                    st.markdown("### ❌ Prediction Errors")
                    error_df = pd.DataFrame(errors)
                    st.dataframe(error_df, use_container_width=True)
                else:
                    st.success("🎉 No prediction errors found in test samples!")
            else:
                st.error("❌ No test results available. Check dataset structure.")

# Tab 4: Model Information
with tab4:
    st.markdown("### 🤖 Model Details")
    st.markdown("""
    **Architecture**: ResNet34  
    **Framework**: FastAI (PyTorch)  
    **Training Data**: Custom dataset with AI-generated and real images  
    **Categories**: animals, city, food, nature, people  
    **Input Size**: Variable (auto-resized to 224×224)  
    **Classes**: 2 (AI-generated, Real)
    """)
    
    st.markdown("### 📈 Training Performance")
    st.markdown("""
    - **Validation Accuracy**: 91.46%
    - **Precision**: 0.92
    - **Recall**: 0.91
    - **F1-Score**: 0.91
    """)
    
    st.markdown("### 🔧 Technical Specifications")
    st.markdown("""
    - **Preprocessing**: Image resizing, normalization
    - **Augmentation**: Random rotations, flips, brightness adjustments
    - **Optimization**: Adam optimizer with learning rate scheduling
    - **Loss Function**: Cross-entropy loss
    """)

# Footer
st.markdown("---")
st.caption("🛠️ Built with Streamlit & FastAI • Model: ResNet34 • Dataset: Custom AI vs Real Images")