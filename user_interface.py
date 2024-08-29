import streamlit as st
import cv2
from real_time_processing import RealTimeProcessing

st.title("Smartphone Photography Enhancement")

cap = cv2.VideoCapture(0)
processor = RealTimeProcessing()

while True:
    ret, frame = cap.read()
    if not ret:
        st.text("Failed to capture video feed.")
        break
    processed_frame = processor.process_image(frame)

    # Convert processed frame to st.image format
    st_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    st.image(st_frame, channels="RGB", use_column_width=True)
    if st.button("Stop"):
        break

cap.release()
