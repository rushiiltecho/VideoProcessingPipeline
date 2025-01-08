import json
import numpy as np
import streamlit as st
import tempfile
import os
import cv2
import pyrealsense2 as rs
import time
from pathlib import Path
from datetime import datetime

from rlef_video_annotation import VideoUploader
from utils import convert_video
from vdeo_analysis_ellm_sudio import VideoAnalyzer
from realsense_recording import RealSenseRecorder
from contextlib import contextmanager
from dsr_control_api.dsr_control_api.cobotclient import CobotClient




def _main():
    st.set_page_config(page_title="Video Stream and Recording UI", layout="wide")

    # Initialize session states
    if 'recorder' not in st.session_state:
        st.session_state.recorder = None
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'run' not in st.session_state:
        st.session_state.run = True
    if 'current_recording_path' not in st.session_state:
        st.session_state.current_recording_path = None

    # Sidebar instructions
    st.sidebar.title("App Settings")
    st.sidebar.markdown(
        """
        1. **Select Mode**: Live RealSense Feed or Upload Video File  
        2. **Follow instructions** based on selected mode  
        3. **View results** (video playback, analysis, annotation, etc.)  
        """
    )
    
    mode = st.sidebar.selectbox(
        "Select Mode",
        ("Live RealSense Feed", "Upload Video File"),
        index=0
    )

    st.title("Video Stream and Recording UI")

    if mode == "Live RealSense Feed":
        handle_live_feed()
    elif mode == "Upload Video File":
        handle_uploaded_file()

def cleanup_existing_cameras():
    """Attempt to clean up any existing RealSense camera connections"""
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()
    except Exception as e:
        st.warning(f"Warning during camera cleanup: {str(e)}")

def _deprecated_initialize_realsense():
    """Initialize RealSense camera with error handling"""
    try:
        cleanup_existing_cameras()
        time.sleep(1)  # Give system time to release resources
        recorder = RealSenseRecorder()
        return recorder, None
    except Exception as e:
        return None, str(e)

def __handle_live_feed():
    """
    Demonstrates a minimal approach for displaying and optionally 
    recording from a live video feed using OpenCV and Streamlit.
    Uses Intel RealSense camera.
    """
    st.subheader("Live Video Feed")
    st.write("Press 'Start Recording' to record and 'Stop Recording' to stop.")

    # Set up RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start the pipeline
    pipeline.start(config)

    recording = False
    output_file = None
    writer = None

    # Columns for the start/stop buttons
    col1, col2 = st.columns(2)

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
                
                frame_width = 640
                frame_height = 480

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
        # Wait for a new frame from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            st.warning("Failed to grab frame from live feed.")
            break

        # Convert RealSense color frame to OpenCV format
        frame = np.asanyarray(color_frame.get_data())

        if recording and writer is not None:
            writer.write(frame)

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(display_frame, channels="RGB")

        # Check if user has clicked the "Stop" button or closed the app
        if not st.session_state["run"]:
            break

    pipeline.stop()

    st.write("Live feed ended. If you recorded, the file is located at:")
    if output_file:
        st.write(output_file.name)

def handle_uploaded_file():
    """Allows a user to upload a file, optionally analyze it, and show the results."""
    st.subheader("Upload and Play Video")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv"])
    if not uploaded_file:
        st.info("Please upload a video file to continue.")
        return

    # Save the uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    original_path = os.path.abspath(uploaded_file.name)
    upload_directory = os.path.dirname(original_path)
    st.write(f"Original file path: {original_path} in {upload_directory}")
    convert_video(video_path, video_path)

    st.success(f"Video uploaded successfully")
    st.video(video_path)

    # Option to analyze
    if st.checkbox("Analyze this video now?", value=True):
        process_saved_recording(video_path)

def process_saved_recording(video_path):
    """Process a recording through video analysis and RLEF upload"""
    progress_bar = st.progress(0)
    
    try:
        # Step 1: Load payload
        st.write("Loading payload...")
        with open("payload.json", "r") as file:
            payload = json.load(file)
        progress_bar.progress(20)

        # Step 2: Upload to GCP and analyze
        st.write("Uploading to GCP and analyzing...")
        analyzer = VideoAnalyzer(payload=payload)
        gcp_url = analyzer.upload_video_to_bucket(
            f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
            video_path
        )
        progress_bar.progress(40)

        # Step 3: Get video annotations
        st.write("Getting video annotations...")
        # annotations = analyzer.get_gemini_response(gcp_url)
        annotations = analyzer.get_ellm_response()
        if annotations:
            st.write("Analysis Complete. Response Annotations:")
            st.json(annotations)
        progress_bar.progress(70)

        # Step 4: Upload to RLEF
        st.write("Uploading to RLEF...")
        rlef_uploader = VideoUploader()
        status = rlef_uploader.upload_to_rlef(
            url="https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/",
            filepath=video_path,
            video_annotations=annotations
        )
        
        progress_bar.progress(100)
        
        if status == 200:
            st.success("Processing completed successfully!")
        else:
            st.warning(f"RLEF upload returned status code: {status}")
            
    except Exception as e:
        st.error(f"Error processing recording: {str(e)}")
        progress_bar.progress(100)



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


class RealSenseManager:
    def __init__(self):
        self.pipeline = None
        self.config = None
        
    def initialize_camera(self):
        """Initialize the RealSense camera with proper resource cleanup"""
        try:
            # First, clean up any existing pipelines
            if self.pipeline:
                self.pipeline.stop()
                self.pipeline = None
            
            # Reset all RealSense devices
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)  # Give devices time to reset
            
            # Initialize new pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Try to start the pipeline
            self.pipeline.start(self.config)
            return True
            
        except Exception as e:
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except:
                    pass
                self.pipeline = None
            st.error(f"Failed to initialize camera: {str(e)}")
            return False
    
    def cleanup(self):
        """Cleanup camera resources"""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None 

def main():
    cobot_client = CobotClient("localhost", 3000)
    #TODO: use this to send the location to Simulation team.
    # cobot_client.known_object_location()
    st.set_page_config(page_title="Video Stream and Recording UI", layout="wide")
    
    if 'run' not in st.session_state:
        st.session_state.run = True
        
    try:
        # Your existing main function code here
        mode = st.sidebar.selectbox(
            "Select Mode",
            ("Upload Video File", "Upload Video File"),
            index=0
        )
        
        if mode == "Live RealSense Feed":
            handle_live_feed()
        elif mode == "Upload Video File":
            handle_uploaded_file()  # Your existing upload handler
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Ensure cleanup happens
        if not st.session_state.get("run", True):
            st.stop()

if __name__ == "__main__":
    main()














# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         st.session_state.run = False
#         if st.session_state.recorder:
#             st.session_state.recorder.stop_recording()
#         cleanup_existing_cameras()
#         st.stop()