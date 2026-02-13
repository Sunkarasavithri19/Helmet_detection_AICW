import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# Page config
st.set_page_config(page_title="Helmet Detection System", layout="wide")

st.title("🪖 Helmet Detection System")
st.markdown("Upload an image to detect **With Helmet** and **Without Helmet**")

# Load model (cached)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # Convert image to RGB (fix 4-channel issue)
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Run detection
    results = model(image_np)

    # Annotated image
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    st.image(annotated_frame, caption="Detection Result", use_container_width=True)

    # Detection summary
    boxes = results[0].boxes
    class_names = model.names

    helmet_count = 0
    no_helmet_count = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        label = class_names[cls_id]

        if label == "With Helmet":
            helmet_count += 1
        else:
            no_helmet_count += 1

    st.subheader("📊 Detection Summary")
    st.write(f"🟢 With Helmet: {helmet_count}")
    st.write(f"🔴 Without Helmet: {no_helmet_count}")
