import os
import json
import librosa
import librosa.display
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
filterwarnings('ignore')


# ==================== CONFIGURAÇÃO DE PÁGINA ====================
def streamlit_config():
    st.set_page_config(page_title='MeloBird', layout='centered', page_icon='🐦')

    st.markdown("""
        <style>
        body {
            background: linear-gradient(180deg, #0f172a, #1e293b);
            color: #f8fafc;
        }
        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }
        h1 {
            text-align: center;
            font-size: 2.2rem;
        }
        .gradient-text {
            background: linear-gradient(90deg, #22d3ee, #a855f7, #f97316);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .upload-box {
            border: 2px dashed #94a3b8;
            padding: 25px;
            border-radius: 15px;
            background-color: rgba(255,255,255,0.05);
        }
        .prediction-card {
            background: rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
            text-align: center;
            transition: all 0.3s ease;
        }
        .prediction-card:hover {
            transform: scale(1.03);
            box-shadow: 0px 0px 20px rgba(255,255,255,0.2);
        }
        .bird-name {
            color: #22d3ee;
            font-size: 1.8rem;
            font-weight: bold;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<h1>🐦 MeloBird 🐦‍⬛<br><span class="gradient-text">Classificação de Sons de Pássaros</span> com Machine Learning</h1>',
        unsafe_allow_html=True
    )
    add_vertical_space(2)


# ==================== FUNÇÃO DE PREDIÇÃO ====================
def prediction(audio_file):
    with open('prediction.json', 'r') as f:
        prediction_dict = json.load(f)

    audio, sample_rate = librosa.load(audio_file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_features = np.mean(mfccs_features, axis=1)
    mfccs_features = np.expand_dims(mfccs_features, axis=(0, 2))

    mfccs_tensors = tf.convert_to_tensor(mfccs_features, dtype=tf.float32)

    model = tf.keras.models.load_model('model.h5')
    prediction = model.predict(mfccs_tensors)

    target_label = np.argmax(prediction)
    predicted_class = prediction_dict[str(target_label)]
    confidence = round(np.max(prediction)*100, 2)

    # --- Exibição do Resultado ---
    add_vertical_space(1)
    st.markdown(f"<h4 style='text-align:center; color:#facc15;'>{confidence:.2f}% de acurácia</h4>", unsafe_allow_html=True)

    image_path = os.path.join('Inference_Images', f'{predicted_class}.jpg')
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (350, 300))

    _, col, _ = st.columns([0.1, 0.8, 0.1])
    with col:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown(f"<div class='bird-name'>{predicted_class}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Exibir espectrograma ---
    st.subheader("Espectrograma do Som")

    fig, ax = plt.subplots(figsize=(8, 3))
    S = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    S_dB = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_dB, sr=sample_rate, x_axis='time', y_axis='mel', cmap='magma', ax=ax
    )
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Espectrograma de Frequências (Mel-Scale)', color='#f8fafc')
    ax.tick_params(colors='#f8fafc')
    plt.tight_layout()

    st.pyplot(fig)


# ==================== INTERFACE STREAMLIT ====================
streamlit_config()

with st.container():
    _, col2, _ = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        input_audio = st.file_uploader("🎵 Faça upload do som de um pássaro (MP3 ou WAV)", type=['mp3', 'wav'])
        st.markdown('</div>', unsafe_allow_html=True)

if input_audio is not None:
    st.audio(input_audio, format='audio/wav')
    with st.spinner('Analisando o canto do pássaro...'):
        prediction(input_audio)
    st.success('Classificação concluída!')
