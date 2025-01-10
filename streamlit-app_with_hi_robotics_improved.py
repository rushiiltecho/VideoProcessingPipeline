from datetime import datetime
import json
import re
from hi_robotics.vision_ai.cameras.intel_realsense_camera import IntelRealSenseCamera
import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import time

from gemini_constant_api_key import GEMINI_API_KEY
from gemini_oop_object_detection import ObjectDetector, demo_flow
from lit_demo_flow import RealSenseManager
from camera_hi_robotics_realsense_pipeline import RealSenseRecorder
from rlef_video_annotation import VideoUploader
from utils import convert_video, get_signed_url, process_images, upload_hdf5_file
from vdeo_analysis_ellm_sudio import VideoAnalyzer
from dsr_control_api.dsr_control_api.cobotclient import CobotClient

output_path = None

def initialize_session_state():
    """Initialize session state variables"""
    if "run" not in st.session_state:
        st.session_state["run"] = True
    if "recording_status" not in st.session_state:
        st.session_state["recording_status"] = False

def create_sidebar():
    """Create an enhanced sidebar with instructions and mode selection"""
    with st.sidebar:
        st.title("🎥 Video Processing")
        
        st.markdown("### 📝 Instructions")
        with st.expander("How to use this app", expanded=True):
            st.markdown("""
            1. **Select Mode** 🔄
               - Live Video Feed: Record from camera
               - Upload Video: Process existing video
            
            2. **Process Video** 🎬
               - Follow on-screen instructions
               - Wait for processing completion
            
            3. **View Results** 📊
               - Check analysis results
               - Download processed data
            """)
        
        mode = st.radio(
            "Select Operating Mode",
            ("Live Video Feed", "Upload Video File"),
            index=1,
            help="Choose how you want to input video data"
        )
        
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        st.success("System Ready")
        
        return mode

def create_header():
    """Create an attractive header section"""
    st.markdown("""
    <h1 style='text-align: center;'>
        Video Stream and Analysis Platform
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; color: gray;'>
        Record, analyze, and process video data with ease
    </p>
    """, unsafe_allow_html=True)

def handle_live_feed():
    """Enhanced live feed handling with better UI feedback"""
    st.subheader("📹 Live Video Feed")
    
    recorder = None
    camera = None
    
    try:
        with st.spinner("🎥 Initializing camera..."):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    camera = IntelRealSenseCamera()
                    recorder = RealSenseRecorder(camera=camera)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        raise e
        
        if not recorder:
            st.error("❌ Failed to initialize camera after multiple attempts")
            return
        
        # Enhanced UI Controls
        st.markdown("### 🎮 Controls")
        controls_col1, controls_col2, controls_col3 = st.columns(3)
        
        with controls_col1:
            if st.button("🟢 Start Recording", use_container_width=True, 
                        disabled=st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = True
                recorder.start_recording('Recorded_Demo')
                st.success(f"📝 Recording to: {recorder.get_current_savepath()}")
                global output_path
                output_path = recorder.get_current_savepath()
        
        with controls_col2:
            if st.button("🔴 Stop Recording", use_container_width=True,
                        disabled=not st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = False
                recorder.stop_recording()
                st.info(f"✅ Captured {recorder.frame_count} frames")
        
        with controls_col3:
            if st.button("⏹️ Quit", use_container_width=True):
                if recorder and recorder.is_recording:
                    recorder.stop_recording()
                st.session_state["run"] = False
                st.rerun()
        
        # Status indicators
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("Recording Status", 
                     "Active 🟢" if st.session_state.get("recording_status", False) else "Inactive 🔴")
        with status_col2:
            if st.session_state.get("recording_status", False):
                st.metric("Frames Captured", recorder.frame_count if recorder else 0)
        
        # Display frames with enhanced layout
        st.markdown("### 📺 Live Preview")
        frame_placeholder = st.empty()
        
        interval = 1/10  # seconds between frames
        last_capture_time = time.time()

        while st.session_state.get("run", True):
            try:
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                depth_colormap = recorder._normalize_depth_for_display(depth_image)
                display_image = np.hstack((color_image, depth_colormap))
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(display_image, channels="RGB", use_container_width=True,
                                    caption="Live Feed (Color + Depth)")
                
                current_time = time.time()
                
                if recorder.is_recording and (current_time - last_capture_time >= interval):
                    recorder._append_frames(rgb_frame, depth_frame)
                    recorder.frame_count += 1

                    with st.spinner("Saving frames..."):
                        rgb_path = f"{recorder.current_savepath}/rgb_images_data_collection/image_{recorder.frame_count}.jpg"
                        cv2.imwrite(rgb_path, color_image)

                        depth_path = f"{recorder.current_savepath}/depth_images_data_collection/image_{recorder.frame_count}.npy"
                        np.save(depth_path, depth_image)

                    last_capture_time = current_time
                    
            except Exception as e:
                st.error(f"❌ Frame capture error: {str(e)}")
                break
                
    except Exception as e:
        st.error(f"❌ Camera initialization failed: {str(e)}")
        return
        
    finally:
        if recorder and recorder.is_recording:
            try:
                recorder.stop_recording()
            except Exception as e:
                st.error(f"❌ Cleanup error: {str(e)}")
        
        if camera:
            try:
                camera.release_camera()
            except Exception as e:
                st.error(f"❌ Camera release error: {str(e)}")
                
        st.session_state["run"] = False

def handle_uploaded_file():
    """Enhanced file upload handling with better UI/UX"""
    st.subheader("📤 Upload and Process Video")
    
    # File upload section with enhanced UI
    upload_col1, upload_col2 = st.columns([1, 1])
    

    uploaded_file = st.file_uploader(
        "Drop your video file here",
        type=["mp4", "avi", "mkv"],
        help="Supported formats: MP4, AVI, MKV"
    )

    # with upload_col2:
    #     st.markdown("### 📋 File Info")
    #     if uploaded_file:
    #         st.success("✅ File uploaded")
    #         st.info(f"📁 Name: {uploaded_file.name}")
    #         st.info(f"📏 Size: {uploaded_file.size / 1024 / 1024:.2f} MB")
    
    if not uploaded_file:
        st.info("👆 Please upload a video file to continue")
        return

    with st.spinner("📝 Processing uploaded file..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        original_path = f"{os.path.abspath(uploaded_file.name)}/recordings"
        upload_directory = os.path.dirname(original_path)
        video_path = tfile.name
        
        convert_video(video_path, video_path)
    
    st.success("✅ Video processed successfully")
    
    # Video preview
    st.markdown("### 🎬 Video Preview")
    st.video(video_path)
    
    # Analysis section
    st.markdown("### 🔍 Analysis Options")
    analyze_video = st.checkbox(
        "Run video analysis",
        value=True,
        help="Perform detailed analysis of the video content"
    )
    
    if analyze_video:
        if st.button("🚀 Start Analysis", use_container_width=True):
            process_saved_recording(video_path)

def process_saved_recording(video_path):
    """Enhanced processing with better progress tracking and UI feedback"""
    st.markdown("### 🔄 Processing Video")
    
    # Create a more detailed progress tracking system
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    result_placeholder = st.empty()
    
    try:
        # Initialize progress and payload
        progress_bar = progress_placeholder.progress(0)
        status_placeholder.markdown("⏳ Initializing process...")
        payload_for_cobot_client = {}
        
        # Load payload
        with st.spinner("📋 Loading configuration..."):
            with open("payload.json", "r") as file:
                payload = json.load(file)
            progress_bar.progress(15)
            status_placeholder.markdown("✅ Configuration loaded")
        
        # Upload and analyze
        analyzer = VideoAnalyzer(payload=payload)
        with st.spinner("☁️ Uploading to cloud..."):
            gcp_url = analyzer.upload_video_to_bucket("test1.mp4", video_path)
            progress_bar.progress(30)
            status_placeholder.markdown("✅ Video uploaded to cloud")
        
        # Get annotations
        with st.spinner("🔍 Analyzing video content..."):
            annotations = analyzer.get_ellm_response()
            if annotations:
                with result_placeholder.expander("📊 View Analysis Results", expanded=True):
                    st.json(annotations)
            progress_bar.progress(45)
            status_placeholder.markdown("✅ Analysis complete")
        
        rlef_uploader = VideoUploader()
        # Upload to RLEF
        with st.spinner("📤 Uploading results to RLEF..."):
            status, rlef_response_text = rlef_uploader.upload_to_rlef(
                rlef_url="https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/",
                video_filepath=video_path,
                video_annotations=annotations,
                csv_filepath=None
            )
            progress_bar.progress(60)
            if status == 200:
                status_placeholder.markdown("✅ Results uploaded to RLEF")
            else:
                st.warning("⚠️ There are issues with RLEF upload")

        # Process coordinates and HAMER predictions
        with st.spinner("📍 Processing coordinates and generating predictions..."):
            recording_dir = 'recordings/Recorded_Demo'
            rgb_zip_path = f'{recording_dir}/archives/rgb_images_data_collection.zip'
            depth_zip_path = f'{recording_dir}/archives/depth_images_data_collection.zip'
            
            # Get coordinates
            detector = ObjectDetector(api_key=GEMINI_API_KEY, recording_dir=recording_dir)
            response_coordinates = detector.get_real_world_coordinates(annotations)
            
            if response_coordinates:
                with result_placeholder.expander("🎯 Coordinate Results", expanded=True):
                    st.json(response_coordinates)
            
            progress_bar.progress(75)
            status_placeholder.markdown("✅ Coordinates processed")
            
            # Prepare payload for cobot client
            payload_for_cobot_client["fundamental_actions"] = {
                key: {
                    **value,
                    "coordinates": value["coordinates"].tolist() if isinstance(value["coordinates"], np.ndarray) else value["coordinates"]
                }
                for key, value in response_coordinates.items()
            }
            payload_for_cobot_client["rlef_resource_id"] = rlef_response_text['_id']
            payload_for_cobot_client["video_gcp_url"] = gcp_url
            
            # Process HAMER predictions

            try:
                csv_hamer_output = process_images(rgb_zip_path, depth_zip_path)
                with open(f"{recording_dir}/hamer_output/predictions_hamer.csv", "r") as file:
                    csv_hamer_output = file.read()
                payload_for_cobot_client["trajectory_csv"] = csv_hamer_output
                status_placeholder.markdown("✅ HAMER predictions loaded")
            except Exception as e:
                st.warning(f"⚠️ Could not load HAMER predictions: {str(e)}")
                payload_for_cobot_client["trajectory_csv"] = ""
            
            progress_bar.progress(90)

        # Send to Cobot Client
        with st.spinner("🤖 Sending data to Cobot..."):
            cobot_client = CobotClient(ip="192.168.0.149", port="8001")
            cobot_client_status = cobot_client.send_trajectory_data(payload_for_cobot_client)
            
            if cobot_client_status:
                status_placeholder.markdown("✅ Data sent to Cobot successfully")
                with result_placeholder.expander("🤖 Cobot Client Payload", expanded=False):
                    st.json(payload_for_cobot_client)
            else:
                st.warning("⚠️ Cobot client response indicates potential issues")
            
        with st.spinner("🤖 Updating the Trajectory in RLEF..."):
            signed_url = get_signed_url(rlef_response_text['_id'], "predictions_hamer.csv")
            if signed_url:
                print(f"Signed URL: {signed_url}")
                upload_hdf5_file(signed_url, f"{recording_dir}/hamer_output/predictions_hamer.csv")
                status_placeholder.markdown("✅ Trajectory updated in RLEF")
            else:
                st.warning("⚠️ Failed to update trajectory in RLEF")
            progress_bar.progress(100)
        
        st.success("🎉 All processing steps completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        progress_placeholder.empty()
        status_placeholder.empty()

def main():
    """Enhanced main function with better UI organization"""
    st.set_page_config(
        page_title="Video Processing Platform",
        page_icon="🎥",
        layout="centered"
    )
    
    initialize_session_state()
    mode = create_sidebar()
    create_header()
    
    if mode == "Live Video Feed":
        handle_live_feed()
    else:
        handle_uploaded_file()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.session_state["run"] = False
        st.stop()