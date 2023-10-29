import io

from PIL import Image, UnidentifiedImageError
import streamlit as st

import image_processing as ip


def loadImageFromUser():
    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is None:
        return None

    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # https://stackoverflow.com/a/32908899
    try:
        image = Image.open(io.BytesIO(bytes_data))
    except UnidentifiedImageError:
        st.write("Uploaded file was not detected as an image.")
        return None

    return image


def renderImage(image):
    st.image(image)


st.header("Game screenshot OCR")
st.write("Upload a screenshot from a game with Japanese text and extract the text!")
image = loadImageFromUser()
if not image:
    exit(0)
debug = st.checkbox("Debug mode")

preprocessedImage = ip.preprocess(image)

croppedImage = ip.cropImage(preprocessedImage)
ocredText = ip.ocr(croppedImage)

st.subheader("Detected text")
st.write(ocredText)

if debug:
    st.subheader("Debug info")
    renderImage(image)
    renderImage(preprocessedImage)
    renderImage(croppedImage)
