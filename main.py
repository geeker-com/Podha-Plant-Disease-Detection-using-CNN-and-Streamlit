
import numpy as np
from tensorflow import keras
from PIL import Image
import streamlit as st
import keras.utils as image
import warnings
import base64
warnings.simplefilter(action='ignore', category=FutureWarning)
import io

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("black.jpg")
logo_img = get_img_as_base64("logo.png")
st.set_page_config(page_title="Podha", page_icon="logo.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://img.freepik.com/free-photo/fresh-green-leaves-grow-sunlight-generated-by-ai_188544-42428.jpg?t=st=1692456172~exp=1692459772~hmac=655f5af4cf10bd0a88285f68a7071d273f1355ce3b41e5cfc8ba7b036aa88a6b&w=1060");
background-size: 100%;
background-position: center;
background-repeat: no-repeat;
background-attachment: local;
background-size:cover;
}}
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

Image_Width = 224
Image_Height = 224
Image_Size = (Image_Width, Image_Height)
classifier = keras.models.load_model('resnet50_plant_disease_final_96.h5', compile=False)

st.markdown("""
    <style>
    .big-font {
        font-size: 40px !important;
        text-align: center;
        font-style: Helvetica;
        font-weight: bold;
        color: white;
        -webkit-text-stroke: 3px black;
        padding: 10px;
    }
    .big-2-font {
        font-size: 40px !important;
        text-align: center;
        color: black;
        font-style: Helvetica;
        font-weight: bold;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Plant Disease Prediction</p>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://www.linkpicture.com/q/garden-plant-in-hand-cartoon-vector-24382370_1_-removebg-preview.png")
    st.markdown("""
    <style>
    .big-font {
        font-size: 40px !important;
        text-align: center;
        color: black;
        font-style: Helvetica;
        font-weight: bold;
    }
    .small-font {
        font-size: 20px !important;
        text-align: center;
        color: purple;
        font-style: Helvetica;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-2-font">Podha</p>', unsafe_allow_html=True)

    st.markdown('<p class="small-font">Developed by<br>Harshit Pokhriyal\n</p>', unsafe_allow_html=True)

results = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        uploaded_image = Image.open(io.BytesIO(image_data))
        resized_image = uploaded_image.resize((150, 150))
        st.image(resized_image, caption='Uploaded Image')
        return uploaded_image


# defining the function which will make the prediction using
# the data which the user inputs
def prediction(im):
    pred = classifier.predict(im)
    print(pred)
    return pred


# this is the main function in which we define our webpage
def main():
    im = load_image()
    if (im != None):
        im = im.resize(Image_Size)
        im = image.img_to_array(im)
        im = np.expand_dims(im, axis=0)
    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    if st.button("Predict"):
        pred = classifier.predict(im)
        class_x = np.argmax(pred, axis=1)
        class_x = int(class_x)
        # print(results[class_x])
        st.success(results[class_x])


if __name__ == '__main__':

    main()
