import streamlit as st
from transformers import ViTFeatureExtractor, BertTokenizer, VisionEncoderDecoderModel
from PIL import Image


def load_image(image_file):
    img = Image.open(image_file).convert("RGB")
    return img

@st.cache
def load_model():
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # El modelo se lee de la carpeta 'model'
    saved_model = VisionEncoderDecoderModel.from_pretrained('model')
    return feature_extractor, tokenizer, saved_model

image_file = st.file_uploader("Choose a file", type=["png","jpg","jpeg"])
feature_extractor, tokenizer, saved_model = load_model()
if image_file is not None:
    bytes_data = image_file.getvalue()

    image = load_image(image_file)
    col1, col2, col3 = st.columns([2, 5, 2])
    col2.image(image, use_column_width=True)

    col1, col2, col3 = st.columns([2, 5, 2])
    btn = col2.button('Crear texto alternativo')
    if btn:    
        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
        pred = saved_model.generate(pixel_values)[0]

        pred_str = tokenizer.decode(pred)
        st.markdown(f"## Predicción")
        st.markdown(pred_str, 4)
        st.markdown("""---""")
        st.markdown(f"## Como texto alternativo")
        pred_processed = pred_str.replace("[CLS]","").replace("[SEP]","").strip()
        code = f'''<img src="{image_file.name}" alt="{pred_processed}">'''
        st.code(code, language='cshtml')
    


    

