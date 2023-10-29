import io

from PIL import Image, UnidentifiedImageError
import streamlit as st

import image_processing as ip


def loadImagesFromUser():
    uploaded_files = st.file_uploader("Upload image(s)", accept_multiple_files=True)
    if len(uploaded_files) == 0:
        exit(0)  # TODO: some sort of error handling?

    for uploaded_file in uploaded_files:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        # https://stackoverflow.com/a/32908899
        try:
            image = Image.open(io.BytesIO(bytes_data))
        except UnidentifiedImageError:
            st.write("Uploaded file was not detected as an image.")
            continue  # TODO: some sort of error handling?

        yield image


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


class OcredImage:
    def __init__(self, image):
        self.original = image

        self.processed = ip.preprocess(image)
        self.cropped = ip.cropImage(self.processed)
        self.text = ip.ocr(self.cropped)

    def renderDebugInfo(self):
        renderImage(self.original)
        renderImage(self.processed)
        renderImage(self.cropped)


def renderMainContent():
    debug = st.checkbox("Debug mode")
    debugImages = []
    for i, image in enumerate(loadImagesFromUser()):
        if not image:
            exit(0)

        if i == 0:
            st.subheader("Detected text")

        ocredImage = OcredImage(image)

        st.write(ocredImage.text)

        if debug:
            debugImages.append(ocredImage)

    if len(debugImages) != 0:
        st.subheader("Debug info")
        for image in debugImages:
            image.renderDebugInfo()


renderHeader()
demo = st.checkbox(label="Demo mode")
if demo:
    renderDemo()
    exit(0)
renderMainContent()
