import json
import streamlit as st
import tempfile
import os
import cv2
import numpy as np

from lit_demo_flow import RealSenseManager
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
    """Handles live feed with improved error handling and resource management"""
    st.subheader("Live Video Feed")
    
    # Initialize camera manager
    camera_manager = RealSenseManager()
    
    if not camera_manager.initialize_camera():
        st.error("Failed to initialize camera. Please ensure no other applications are using it and try again.")
        return
    
    try:
        # Recording state
        recording = False
        writer = None
        output_file_path = None
        
        # UI Controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Recording"):
                if not recording:
                    recording = True
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fd, output_file_path = tempfile.mkstemp(suffix='.mp4')
                    os.close(fd)
                    
                    writer = cv2.VideoWriter(
                        output_file_path,
                        fourcc,
                        20.0,
                        (640, 480)
                    )
                    st.success(f"Recording started. Output file: {output_file_path}")
        
        with col2:
            if st.button("Stop Recording"):
                if recording and writer:
                    recording = False
                    writer.release()
                    writer = None
                    st.info("Recording stopped")
        
        # Display frame
        stframe = st.empty()
        
        while True:
            try:
                # Get frames with timeout
                frames = camera_manager.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
                
                # Record if active
                if recording and writer:
                    writer.write(frame)
                
                # Display frame
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(display_frame, channels="RGB")
                
                # Check for stop condition
                if not st.session_state.get("run", True):
                    break
                    
            except Exception as e:
                st.error(f"Error during frame capture: {str(e)}")
                break
                
    finally:
        # Cleanup
        if writer:
            writer.release()
        camera_manager.cleanup()
        
        if output_file_path and os.path.exists(output_file_path):
            st.write(f"Recording saved to: {output_file_path}")

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
