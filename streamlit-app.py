import streamlit as st
import tempfile
import os

def main():
    st.title("Video Stream and Recording UI")

    mode = st.selectbox("Select Mode", ("Live Video Feed", "Upload Video File"))

    if mode == "Live Video Feed":
        handle_live_feed()
    elif mode == "Upload Video File":
        handle_uploaded_file()

def handle_live_feed():
    st.subheader("Live Video Feed")

    import cv2
    cap = cv2.VideoCapture(0)
    recording = [False]
    output_file = None

    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Recording")
    with col2:
        stop_button = st.button("Stop Recording")

    if start_button:
        if not recording[0]:
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            recording[0] = True
            st.success(f"Recording started. File: {output_file.name}")
    elif stop_button:
        if recording[0]:
            recording[0] = False
            st.info("Recording stopped.")

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from live feed.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()

def handle_uploaded_file():
    st.subheader("Upload and Play Video")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv"])
    print(uploaded_file)
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.success(f"Video uploaded successfully: {video_path}")

        st.video(video_path)

if __name__ == "__main__":
    main()
