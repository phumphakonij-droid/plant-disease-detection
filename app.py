import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# โหลดโมเดล
model = tf.keras.models.load_model('plant_disease_model.keras')

# ชื่อคลาสทั้ง 15
class_names = [
    'Pepper__bell__Bacterial_spot', 'Pepper__bell__healthy',
    'Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

st.title("🌿 Plant Disease Detection")
st.write("อัปโหลดรูปใบพืชเพื่อตรวจโรค")

uploaded_file = st.file_uploader("เลือกรูป", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, caption="รูปที่อัปโหลด")
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.success(f"ผลลัพธ์: **{result}**")
    st.info(f"ความมั่นใจ: **{confidence:.2f}%**")
