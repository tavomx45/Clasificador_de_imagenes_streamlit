import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image

# Configuración de página
st.set_page_config(page_title="Clasificador de Imágenes", page_icon="🖼️")
st.title("🖼️ Clasificador de Imágenes con MobileNetV2")
st.write("Carga una imagen y el modelo clasificará lo que observa usando pesos de ImageNet.")

# Cargar el modelo preentrenado directamente sin API Keys
@st.cache_resource
def load_model():
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()  # Modo evaluación
    return model, weights

model, weights = load_model()
preprocess = weights.transforms()
categories = weights.meta["categories"]

# Interfaz para subir imagen
uploaded_file = st.file_uploader("Elige una imagen (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen seleccionada", use_container_width=True)
    
    if st.button("Clasificar Imagen"):
        with st.spinner("Analizando la imagen..."):
            # Preprocesar imagen para el modelo
            input_tensor = preprocess(image).unsqueeze(0)
            
            # Hacer la predicción
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Obtener el resultado con mayor probabilidad
            top_prob, top_catid = torch.topk(probabilities, 1)
            category = categories[top_catid[0]]
            confidence = top_prob[0].item() * 100
            
            st.success(f"**Predicción:** {category.capitalize()}")
            st.metric("Nivel de Confianza", f"{confidence:.2f}%")
