# ============================================================
# PART 1: CNN MODEL TRAINING (from the Colab notebook)
# ============================================================

import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
import os

MODEL_PATH = "mnist_cnn.keras"

# Train and save the CNN only if the trained model file is missing.
if not os.path.exists(MODEL_PATH):
    print("Saved model not found. Training CNN model...")

    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

    # Build the CNN architecture
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))

    # Save the trained model
    model.save(MODEL_PATH)
    print("CNN model trained and saved.")
else:
    print("Saved CNN model found. Skipping training.")


# ============================================================
# PART 2: STREAMLIT APP (user's original code, unchanged)
# ============================================================

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(
    page_title="CNN Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

# Load trained CNN model
model = tf.keras.models.load_model("mnist_cnn.keras")

st.title("🔢 Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0–9) and let the CNN predict it.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Convert to grayscale
    gray_image = image.convert("L")

    # Resize to 28 x 28
    resized_image = gray_image.resize((28, 28))

    st.subheader("Original Image")
    st.image(image)

    st.subheader("Grayscale Image")
    st.image(gray_image)

    st.subheader("Resized Image (28 x 28)")
    st.image(resized_image)

    # Convert image into NumPy array
    image_array = np.array(resized_image)

    st.subheader("Pixel Values")
    st.text(str(image_array))

    st.text(f"Shape: {image_array.shape}")
    st.text(f"Minimum pixel value: {image_array.min()}")
    st.text(f"Maximum pixel value: {image_array.max()}")

    # Normalize pixel values
    normalized_image = image_array / 255.0

    st.subheader("Normalized Pixel Values")
    st.text(str(normalized_image))

    st.text(f"Normalized Minimum: {normalized_image.min()}")
    st.text(f"Normalized Maximum: {normalized_image.max()}")

    # Automatically detect background and invert if necessary
    if image_array.mean() > 127:
        inverted_image = 1.0 - normalized_image
    else:
        inverted_image = normalized_image

    st.subheader("Processed Image for CNN")
    st.image(inverted_image)


    # Prepare image for CNN input
    cnn_input = inverted_image.reshape(1, 28, 28, 1)

    st.subheader("CNN Input Shape")
    st.text(str(cnn_input.shape))

    # Make prediction
    prediction = model.predict(cnn_input)

    # Get predicted digit
    predicted_digit = np.argmax(prediction)

    # Get confidence score
    confidence = np.max(prediction) * 100

    st.subheader("Prediction Result")
    st.metric("Predicted Digit", int(predicted_digit))
    st.metric("Confidence", f"{confidence:.2f}%")
