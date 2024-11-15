import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json

st.set_page_config(layout='wide')
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
    st.write(cache['A3'].shape)
    predictions = get_predictions(cache[f'A{len(layers)-1}'])
    return predictions.flatten()

def preprocessing(image, target_size):
    resized_image = cv2.resize(image, target_size)
    normalized_image = resized_image / 255.0
    flattened_image = normalized_image.flatten()

    return flattened_image.reshape(-1, 1)

with open('params.json', 'r') as json_file:
    params = json.load(json_file)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    image = np.array(pil_image)
    st.write(image.shape)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    preprocessed_image = preprocessing(image, target_size=(128, 36))
    st.write(preprocessed_image.T.shape)
    predict_res = predict(preprocessed_image, params, layers=(len(preprocessed_image), 128, 128, 1))

    st.write(predict_res)

