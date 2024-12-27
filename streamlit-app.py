import json
import streamlit as st
import tempfile
import os
import cv2

from rlef_video_annotation import VideoUploader
from vdeo_analysis_ellm_sudio import VideoAnalyzer

def main():
    st.set_page_config(page_title="Video Stream and Recording UI", layout="centered")

    # Sidebar instructions or information
    st.sidebar.title("App Settings")
    st.sidebar.markdown(
        """
        1. **Select Mode**: Live Video Feed or Upload Video File  
        2. **Follow instructions** based on selected mode  
        3. **View results** (video playback, analysis, annotation, etc.)  
        """
    )
    
    mode = st.sidebar.selectbox(
        "Select Mode",
        ("Live Video Feed", "Upload Video File"),
        index=1
    )

    st.title("Video Stream and Recording UI")

    if mode == "Live Video Feed":
        handle_live_feed()
    elif mode == "Upload Video File":
        handle_uploaded_file()

def handle_live_feed():
    """
    Demonstrates a minimal approach for displaying and optionally 
    recording from a live video feed using OpenCV and Streamlit.
    """
    st.subheader("Live Video Feed")
    st.write("Press 'Start Recording' to record and 'Stop Recording' to stop.")

    # Attempt to open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam. Please ensure a webcam is connected.")
        return

    recording = False
    output_file = None
    writer = None

    # Columns for the start/stop buttons
    col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("Start Recording"):
    #         if not recording:
    #             # Create a temp file for saving recorded video
    #             output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    #             st.success(f"Recording started. Output file: {output_file.name}")
    #             recording = True

    #             # Set up video writer to save the feed
    #             # Adjust fps, frame size, and fourcc as needed
    #             fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #             frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #             frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #             writer = cv2.VideoWriter(
    #                 output_file.name,
    #                 fourcc,
    #                 20.0,  # frames per second
    #                 (frame_width, frame_height),
    #             )

    with col1:
        if st.button("Start Recording"):
            if not recording:
                recording = True

                # 1) Use a consistent FourCC + extension
                fourcc = cv2.VideoWriter_fourcc(*"XVID")  # or "MJPG", "avc1", etc.
                
                # 2) Generate a unique .avi path, no open file handle
                fd, path = tempfile.mkstemp(suffix=".mp4")
                os.close(fd)  # release the file descriptor immediately
                output_file_path = path

                st.success(f"Recording started. Output file: {output_file_path}")
                
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                writer = cv2.VideoWriter(
                    output_file_path,
                    fourcc,
                    20.0,  # FPS
                    (frame_width, frame_height),
                )

    with col2:
        if st.button("Stop Recording"):
            if recording:
                recording = False
                if writer is not None:
                    writer.release()
                    writer = None
                st.info("Recording stopped.")

    stframe = st.empty()
    
    # Display the live feed in real-time
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame from live feed.")
            break

        if recording and writer is not None:
            writer.write(frame)

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(display_frame, channels="RGB")

        # Check if user has clicked the "Stop" button or closed the app
        if not st.session_state["run"]:
            break

        # Optionally remove or replace this safeguard:
        # if not st.query_params:  # if you want to check emptiness
        #     break

    cap.release()
    if writer is not None:
        writer.release()

    st.write("Live feed ended. If you recorded, the file is located at:")
    if output_file:
        st.write(output_file.name)

def handle_uploaded_file():
    """Allows a user to upload a file, optionally analyze it, and show the results."""
    st.subheader("Upload and Play Video")

    # Provide a file uploader
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv"])
    if not uploaded_file:
        st.info("Please upload a video file to continue.")
        return

    # Save the uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.success(f"Video uploaded successfully: {video_path}")
    st.video(video_path)

    # Optionally let the user decide whether to analyze now or not
    analyze_now = st.checkbox("Analyze this video now?", value=True)

    if analyze_now:
        process_uploaded_video(video_path)

def process_uploaded_video(video_path):
    """
    Handles uploading the video to GCP, analyzing it with ELLM/Gemini,
    then uploading annotations to RLEF.
    """
    # Progress bar to show the user we're doing background tasks
    progress_bar = st.progress(0)
    
    # Step 1: Load payload from JSON
    st.write("Loading payload...")
    try:
        with open("payload.json", "r") as file:
            payload = json.load(file)
    except Exception as e:
        st.error(f"Error reading payload.json: {e}")
        return

    progress_bar.progress(10)

    # Step 2: Instantiate the analyzer and upload the video to GCP
    st.write("Uploading video to GCP...")
    analyzer = VideoAnalyzer(payload=payload)
    try:
        gcp_url = analyzer.upload_video_to_bucket("test1.mp4", video_path)
    except Exception as e:
        st.error(f"Error uploading to GCP bucket: {e}")
        return

    st.write(f"Uploaded to GCP. URL: {gcp_url}")
    progress_bar.progress(40)

    # Step 3: Get response annotations from the analyzer
    st.write("Analyzing video with Gemini/ELLM...")
    response_annotations = None
    with st.spinner("Generating response..."):
        try:
            response_annotations = analyzer.get_gemini_response(gcp_url=gcp_url)
        except Exception as e:
            st.error(f"Error analyzing video: {e}")
            return

    if response_annotations:
        st.write("Analysis Complete. Response Annotations:")
        st.json(response_annotations)
    else:
        st.warning("No response annotations found.")
    
    progress_bar.progress(70)

    # Step 4: Upload annotations to RLEF
    st.write("Uploading annotations to RLEF tool...")
    rlef_annotations = VideoUploader()
    try:
        status = rlef_annotations.upload_to_rlef(
            url="https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/",
            filepath=video_path,
            video_annotations=response_annotations
        )
        st.write(f"RLEF Annotated Data Upload Status Code: {status}")
    except Exception as e:
        st.error(f"Error uploading annotations to RLEF: {e}")
        return

    progress_bar.progress(100)
    st.success("All steps completed successfully!")


# Use session state to help with stop/clean mechanism
if "run" not in st.session_state:
    st.session_state["run"] = True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.session_state["run"] = False
        st.stop()
