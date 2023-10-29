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


def renderHeader():
    st.header("Game screenshot OCR")
    st.write("Upload a screenshot from a game with Japanese text and extract the text!")


def renderDemo():
    image = Image.open("files/ff4_sample.jpg")
    preprocessedImage = ip.preprocess(image)
    croppedImage = ip.cropImage(preprocessedImage)
    ocredText = ip.ocr(croppedImage)

    st.subheader("Target image")
    renderImage(image)

    st.subheader("Detected text")
    st.write(ocredText)


def renderMainContent():
    debug = st.checkbox("Debug mode")
    image = loadImageFromUser()
    if not image:
        exit(0)

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


renderHeader()
demo = st.checkbox(label="Demo mode")
if demo:
    renderDemo()
    exit(0)
renderMainContent()
