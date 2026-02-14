import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# Page config
st.set_page_config(page_title="Helmet Detection System", layout="wide")

st.title("🪖 Helmet Detection System")
st.markdown("Upload an image to detect **With Helmet** and **Without Helmet**")

# Create two columns
col1, col2 = st.columns([3, 1])  # Left bigger, Right smaller

# RIGHT SIDE - Developer Info
with col2:
    st.markdown("### 👨‍💻 Developed By")
    st.markdown("**M Satya**")
    st.markdown("**S Savithri**")

    st.markdown("---")
    st.markdown("### 🎓 Guide")
    st.markdown("**K**")

# Load model (cached)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# LEFT SIDE - Main App
with col1:
    uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        # Convert image to RGB (no resizing for detection)
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Show small preview while analyzing
        st.image(image, caption="Uploaded Image", width=250)

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

        if boxes is not None:
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

