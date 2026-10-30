import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Pencil Sketch Generator",
    page_icon="✏️",
    layout="wide"
)

st.title("✏️ Image to Pencil Sketch Converter")
st.write("Upload an image and convert it into a pencil sketch using OpenCV.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read uploaded image
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Unable to read the uploaded image.")
    else:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Sketch settings
        st.sidebar.header("Sketch Settings")

        blur_size = st.sidebar.slider(
            "Blur kernel size",
            min_value=11,
            max_value=101,
            value=81,
            step=10
        )

        if blur_size % 2 == 0:
            blur_size += 1

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Invert grayscale image
        inverted = 255 - gray

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(
            inverted,
            (blur_size, blur_size),
            0
        )

        # Invert blurred image
        inverted_blur = 255 - blur

        # Generate pencil sketch
        sketch = cv2.divide(
            gray,
            inverted_blur,
            scale=256
        )

        # Display images
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(img_rgb, use_container_width=True)

        with col2:
            st.subheader("Pencil Sketch")
            st.image(
                sketch,
                clamp=True,
                use_container_width=True
            )

        # Convert sketch to downloadable JPEG
        success, encoded_image = cv2.imencode(
            ".jpg", sketch
        )

        if success:
            st.download_button(
                label="Download Pencil Sketch",
                data=encoded_image.tobytes(),
                file_name="pencil_sketch.jpg",
                mime="image/jpeg"
            )

        st.success("Pencil sketch generated successfully!")