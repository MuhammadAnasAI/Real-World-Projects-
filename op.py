import os
# Configure environment variables first
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set page config
st.set_page_config(
    page_title="Rice Disease Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .stFileUploader>div>div>div>div {
        color: #4CAF50;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    h1 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = []
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

@st.cache_data
def load_and_preprocess_data(data_path='train_images', csv_path='train.csv'):
    """Load and preprocess the dataset with error handling"""
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        
        required_columns = ['image_id', 'label', 'variety']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV missing required columns: {required_columns}")
        
        # Initialize and store label encoder
        st.session_state.label_encoder = LabelEncoder()
        df['label_encoded'] = st.session_state.label_encoder.fit_transform(df['label'])
        df['variety_encoded'] = LabelEncoder().fit_transform(df['variety'])
        
        # Create image paths with proper string conversion
        df['image_path'] = df.apply(
            lambda row: os.path.join(str(data_path), str(row['label']), str(row['image_id'])),
            axis=1
        )
        
        # Verify at least one image exists
        sample_path = df.iloc[0]['image_path']
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample image not found: {sample_path}")
        
        # Split data with stratification
        train_df, val_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['label_encoded']
        )
        
        return train_df, val_df, st.session_state.label_encoder.classes_
    
    except Exception as e:
        st.error(f"Error in data loading: {str(e)}")
        return None, None, None

def create_data_generators(train_df, val_df, img_size=(224, 224), batch_size=32):
    """Create data generators with augmentation"""
    try:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='image_path',
            y_col='label_encoded',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='raw'
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            x_col='image_path',
            y_col='label_encoded',
            target_size=img_size,
            batch_size=batch_size,
            class_mode='raw'
        )
        
        return train_generator, val_generator
    except Exception as e:
        st.error(f"Error creating data generators: {str(e)}")
        return None, None

def build_model(num_classes, img_size=(224, 224, 3)):
    """Build EfficientNetB3 model with transfer learning"""
    try:
        base_model = applications.EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=img_size,
            pooling='avg'
        )
        base_model.trainable = False
        
        inputs = layers.Input(shape=img_size)
        x = base_model(inputs, training=False)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        st.error(f"Error building model: {str(e)}")
        return None

def train_model(model, train_generator, val_generator, epochs=2):
    """Train model with callbacks"""
    try:
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ]
        
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        return None

def plot_training_history(history):
    """Plot training and validation metrics"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error plotting history: {str(e)}")

def display_sample_images():
    """Display sample images from the dataset"""
    try:
        sample_images = [
            'train_images/normal/109760.jpg',
            'train_images/dead_heart/105159.jpg',
            'train_images/blast/110243.jpg',
            'train_images/bacterial_leaf_blight/109372.jpg'
        ]
        
        cols = st.columns(4)
        for idx, img_path in enumerate(sample_images):
            if os.path.exists(img_path):
                img = Image.open(img_path)
                cols[idx].image(img, caption=os.path.basename(img_path), width=150)
            else:
                st.warning(f"Image not found: {img_path}")
    except Exception as e:
        st.error(f"Error displaying samples: {str(e)}")

def main():
    st.title("🍚 Rice Disease Classification System")
    
    menu = ["Home", "Train Model", "Predict", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.header("Welcome to Rice Disease Classifier")
        st.write("""
        This application helps identify various rice diseases using deep learning.
        You can:
        - Train a new model
        - Make predictions on new images
        - View model performance
        """)
        
        if st.button("View Sample Images"):
            display_sample_images()
    
    elif choice == "Train Model":
        st.header("Train a New Model")
        
        with st.expander("Data Information", expanded=True):
            if os.path.exists('train.csv'):
                try:
                    df = pd.read_csv('train.csv')
                    st.write("Dataset Overview:")
                    st.dataframe(df.head())
                    
                    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                    sns.countplot(data=df, x='label', ax=ax[0])
                    ax[0].set_title('Disease Distribution')
                    ax[0].tick_params(axis='x', rotation=45)
                    
                    sns.countplot(data=df, x='variety', ax=ax[1])
                    ax[1].set_title('Variety Distribution')
                    ax[1].tick_params(axis='x', rotation=45)
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        if st.button("Start Training", key="train_button"):
            with st.spinner("Loading and preprocessing data..."):
                train_df, val_df, class_names = load_and_preprocess_data()
                if train_df is not None:
                    st.session_state.class_names = class_names
                    train_generator, val_generator = create_data_generators(train_df, val_df)
                    
                    if train_generator:
                        st.success(f"Data loaded successfully! Found {len(class_names)} classes.")
                        st.write("Class names:", class_names)
                        
                        with st.spinner("Building and training model..."):
                            model = build_model(len(class_names))
                            if model:
                                st.session_state.model = model
                                history = train_model(model, train_generator, val_generator)
                                
                                if history:
                                    st.session_state.history = history
                                    st.success("Training completed!")
                                    plot_training_history(history)
                                    
                                    # Save model
                                    try:
                                        model.save('rice_disease_model.h5')
                                        st.success("Model saved as 'rice_disease_model.h5'")
                                    except Exception as e:
                                        st.error(f"Error saving model: {str(e)}")
    
    elif choice == "Predict":
        st.header("Make Predictions")
        
        if st.session_state.model is None:
            if os.path.exists('rice_disease_model.h5'):
                try:
                    st.session_state.model = models.load_model('rice_disease_model.h5')
                    st.success("Loaded pre-trained model!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
            else:
                st.warning("Please train a model first or load a pre-trained model.")
        
        uploaded_file = st.file_uploader(
            "Upload a rice leaf image",
            type=['jpg', 'jpeg', 'png'],
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", width=300)
                
                if st.button("Predict", key="predict_button"):
                    if st.session_state.model is not None and st.session_state.label_encoder is not None:
                        # Preprocess image
                        img = img.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Make prediction
                        prediction = st.session_state.model.predict(img_array)
                        predicted_class = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        st.success(
                            f"Prediction: {st.session_state.class_names[predicted_class]} "
                            f"(Confidence: {confidence:.2%})"
                        )
                        
                        # Show prediction probabilities
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.barh(st.session_state.class_names, prediction[0])
                        ax.set_xlabel('Probability')
                        ax.set_title('Prediction Probabilities')
                        st.pyplot(fig)
                    else:
                        st.error("Model or label encoder not properly initialized!")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    elif choice == "About":
        st.header("About")
        st.write("""
        **Rice Disease Classification System**
        
        This application uses deep learning to classify various rice diseases including:
        - Bacterial Leaf Blight
        - Blast
        - Brown Spot
        - Dead Heart
        - Hispa
        - Normal leaves
        
        The model is based on EfficientNetB3 architecture with transfer learning.
        """)
        st.write("Developed for agricultural disease detection and prevention.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")