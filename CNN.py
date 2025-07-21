import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import cv2
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set page config
st.set_page_config(page_title="Rice Disease Classifier", layout="wide")

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
    }
    .stFileUploader>div>div>div>div {
        color: #4CAF50;
    }
    .css-1aumxhk {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = []

# Data Preparation Functions
@st.cache_data
def load_and_preprocess_data(data_path='train_images', csv_path='train.csv'):
    df = pd.read_csv(csv_path)
    
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    df['variety'] = le.fit_transform(df['variety'])
    
    # Create image paths
    df['image_path'] = df.apply(lambda row: os.path.join(data_path, row['label'], row['image_id']), axis=1)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    return train_df, val_df, le.classes_

def create_data_generators(train_df, val_df, img_size=(224, 224), batch_size=32):
    # Data augmentation for training
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
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='raw'
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='label',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='raw'
    )
    
    return train_generator, val_generator

# Model Functions
def build_model(num_classes, img_size=(224, 224, 3)):
    base_model = applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=img_size,
        pooling='avg'
    )
    
    # Freeze base model layers
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

def train_model(model, train_generator, val_generator, epochs=20):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

# Streamlit UI
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
    
    elif choice == "Train Model":
        st.header("Train a New Model")
        
        with st.expander("Data Information"):
            if os.path.exists('train.csv'):
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
        
        if st.button("Start Training"):
            with st.spinner("Loading and preprocessing data..."):
                train_df, val_df, class_names = load_and_preprocess_data()
                st.session_state.class_names = class_names
                
                train_generator, val_generator = create_data_generators(train_df, val_df)
                
                st.success(f"Data loaded successfully! Found {len(class_names)} classes.")
                st.write("Class names:", class_names)
            
            with st.spinner("Building and training model..."):
                model = build_model(len(class_names))
                st.session_state.model = model
                
                history = train_model(model, train_generator, val_generator)
                st.session_state.history = history
                
                st.success("Training completed!")
                
                # Plot training history
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
                
                # Save model
                model.save('rice_disease_model.h5')
                st.success("Model saved as 'rice_disease_model.h5'")
    
    elif choice == "Predict":
        st.header("Make Predictions")
        
        if st.session_state.model is None:
            if os.path.exists('rice_disease_model.h5'):
                st.session_state.model = models.load_model('rice_disease_model.h5')
                st.success("Loaded pre-trained model!")
            else:
                st.warning("Please train a model first or load a pre-trained model.")
        
        uploaded_file = st.file_uploader("Upload a rice leaf image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", width=300)
            
            # Preprocess image
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            if st.button("Predict"):
                if st.session_state.model is not None:
                    prediction = st.session_state.model.predict(img_array)
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    st.success(f"Prediction: {st.session_state.class_names[predicted_class]} (Confidence: {confidence:.2%})")
                    
                    # Show prediction probabilities
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(st.session_state.class_names, prediction[0])
                    ax.set_xlabel('Probability')
                    ax.set_title('Prediction Probabilities')
                    st.pyplot(fig)
                else:
                    st.error("No model available for prediction!")
    
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
    main()