import io

from PIL import Image, UnidentifiedImageError
import streamlit as st

import image_processing as ip


def loadImagesFromUser(uploadedFiles):
    for uploaded_file in uploadedFiles:
        # To read file as bytes:
        bytesData = uploaded_file.getvalue()

        # https://stackoverflow.com/a/32908899
        try:
            image = Image.open(io.BytesIO(bytesData))
        except UnidentifiedImageError:
            st.write("Uploaded file was not detected as an image.")
            continue  # TODO: some sort of error handling?

        yield image


def renderImage(image):
    st.image(image)


def renderHeader():
    st.header("Game screenshot OCR")
    st.write("Upload a screenshot from a game with Japanese text and extract the text!")
    st.write(
        "(Currently this probably won't work well for games other than Final Fantasy 4)"
    )


def renderFooter():
    st.caption("[Github source](https://github.com/timmahrt/rpg_img_to_text)")


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
    uploadedFiles = st.file_uploader("Upload image(s)", accept_multiple_files=True)
    if len(uploadedFiles) == 0:
        return

    for i, image in enumerate(loadImagesFromUser(uploadedFiles)):
        if not image:
            return

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
else:
    renderMainContent()
renderFooter()
