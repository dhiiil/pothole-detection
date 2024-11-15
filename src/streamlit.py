import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
import os

# st.set_page_config(layout='wide')
st.title('🛣️ Pothole Detection 🛣️')

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

def sig(X):
    return 1 / (1 + np.exp(-X)) 

def relu(X):
    return np.maximum(X, 0)

def feed_forward(X, params, layers):
    cache = {'A0': X}
    for i in range(1, len(layers)):
        Z = np.dot(params[f'W{i}'], cache[f'A{i-1}']) + params[f'b{i}']
        if i == len(layers) - 1:
            A = sig(Z)
        else:
            A = relu(Z)
        cache[f'Z{i}'] = Z
        cache[f'A{i}'] = A

    return cache

def get_predictions(X):
    return (X > 0.5).astype(int)

def predict(X_test, params, layers):
    cache = feed_forward(X_test, params, layers)
    predictions = get_predictions(cache[f'A{len(layers)-1}'])
    return predictions.flatten()

def preprocessing(image, target_size):
    resized_image = cv2.resize(image, target_size)
    normalized_image = resized_image / 255.0
    flattened_image = normalized_image.flatten()

    return flattened_image.reshape(-1, 1)

params_file_path = os.path.join(os.path.dirname(__file__), 'params.json')

# Periksa apakah file ada
if os.path.exists(params_file_path):
    with open(params_file_path, 'r') as json_file:
        params = json.load(json_file)
else:
    st.error(f"File '{params_file_path}' tidak ditemukan. Pastikan file tersebut ada di server.")
    st.stop()

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    image = np.array(pil_image)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    preprocessed_image = preprocessing(image, target_size=(128, 36))
    st.header('Hasil Prediksi :')
    predict_res = predict(preprocessed_image, params, layers=(len(preprocessed_image), 128, 128, 1))

    class_predict = 'Jalan Berlubang Moas' if predict==1 else 'Alhamdulillah Jalan Normal'
    st.write(class_predict)

