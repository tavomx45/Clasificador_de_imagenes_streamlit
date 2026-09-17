import streamlit as st
import kagglehub
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuración de la interfaz
st.set_page_config(page_title="Clasificador con Kaggle Models", page_icon="🖼️")
st.title("Clasificador de Imágenes con Kaggle Models Estudiante: Bryan Gustavo Paredes")
st.write("Carga una imagen y clasifícala usando el modelo preentrenado MobileNetV2 de Kaggle.")


@st.cache_resource
def load_kaggle_model():
    # Descarga la última versión del modelo MobileNetV2 desde Kaggle
    model_path = kagglehub.model_download("google/mobilenet-v2/tfLite/100-224-feature-vector")
    return model_path

st.info("Descargando/Cargando modelo desde Kaggle Models...")
model_path = load_kaggle_model()
st.success("¡Modelo cargado correctamente desde Kaggle Hub!")

@st.cache_data
def get_labels():
    labels_path = tf.keras.utils.get_file(
        'ImageNetLabels.txt',
        'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    )
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = get_labels()


uploaded_file = st.file_uploader("Elige una imagen (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    if st.button("Clasificar Imagen"):
        with st.spinner("Analizando la imagen..."):
            # Preprocesamiento para MobileNetV2 (224x224)
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
            
            # Cargar y ejecutar con TFLite
            interpreter = tf.lite.Interpreter(model_path=f"{model_path}/1.tflite")
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_index = np.argmax(output_data[0])
            
            # Resultado
            predicted_label = labels[predicted_index]
            confidence = np.max(tf.nn.softmax(output_data[0])) * 100
            
            st.success(f"**Predicción:** {predicted_label.capitalize()}")
            st.metric("Nivel de confianza", f"{confidence:.2f}%")
