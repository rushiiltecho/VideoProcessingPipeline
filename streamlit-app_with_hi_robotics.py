import os
import time
import json
import cv2
import numpy as np
import pandas as pd
import tempfile
import streamlit as st

from PIL import Image
from datetime import datetime
from hi_robotics.vision_ai.cameras.intel_realsense_camera import IntelRealSenseCamera

from gemini_constant_api_key import GEMINI_API_KEY
from gemini_oop_object_detection import ObjectDetector, demo_flow
from lit_demo_flow import RealSenseManager
from camera_hi_robotics_realsense_pipeline import RealSenseRecorder
from model.model import predict_trajectory, save_predictions_to_csv
from rlef_video_annotation import VideoUploader
from utils import (
    convert_video,
    get_real_world_coordinates,
    get_signed_url,
    process_images,
    transform_coordinates,
    upload_hdf5_file,
)
from vdeo_analysis_ellm_sudio import VideoAnalyzer
from dsr_control_api.dsr_control_api.cobotclient import CobotClient

###############################################################################
#                       CONFIGURATION SECTION
###############################################################################

def configure_paths():
    """
    Create a sidebar section for configuring filepaths so that no paths are hardcoded.
    Adjust default values as needed for your own environment.
    """
    with st.sidebar.expander("⚙️ Path Configuration", expanded=False):
        base_dir = st.text_input("Base Recording Directory:", value="recordings")
        rec_name = st.text_input("Recording Subdirectory:", value="Recorded_Demo")
        model_path = st.text_input("Model Path (.pth):", value="model/pouring_trajectory_model.pth")
        payload_path = st.text_input("Payload JSON Path:", value="payload.json")

        if "base_dir" not in st.session_state:
            st.session_state["base_dir"] = base_dir
        if "rec_name" not in st.session_state:
            st.session_state["rec_name"] = rec_name
        if "model_path" not in st.session_state:
            st.session_state["model_path"] = model_path
        if "payload_path" not in st.session_state:
            st.session_state["payload_path"] = payload_path

        # Update session state if user changes text inputs
        if st.button("Save Path Configuration"):
            st.session_state["base_dir"] = base_dir
            st.session_state["rec_name"] = rec_name
            st.session_state["model_path"] = model_path
            st.session_state["payload_path"] = payload_path
            st.success("✅ File paths updated in session state!")

###############################################################################
#                          GLOBAL STATE AND HELPERS
###############################################################################

output_path = None

def initialize_session_state():
    """Initialize session state variables (e.g. for controlling camera feed)."""
    if "run" not in st.session_state:
        st.session_state["run"] = True
    if "recording_status" not in st.session_state:
        st.session_state["recording_status"] = False

def create_sidebar():
    """Create a sidebar with instructions and mode selection."""
    with st.sidebar:
        st.title("🎥 Video Processing")
        
        st.markdown("### 📝 Instructions")
        with st.expander("How to use this app", expanded=True):
            st.markdown("""
            1. **Select Mode** 🔄
               - Live Video Feed: Record from camera
               - 8-Second Recording: Record short timed video
               - Upload Video File: Process existing video
               - Run Inference: Pick frames and run object-detection/trajectory logic

            2. **Process Video** 🎬
               - Follow on-screen instructions
               - Wait for processing completion

            3. **View Results** 📊
               - Check analysis results
               - Download processed data
            """)
        
        mode = st.radio(
            "Select Operating Mode",
            ("Live Video Feed", '8-Second Recording', "Upload Video File", "Run Inference"),
            index=2,
            help="Choose how you want to input video data"
        )
        
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        st.success("System Ready")
        
        return mode


def create_header():
    """Create a header section for the app."""
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

###############################################################################
#                          VIDEO HANDLING FUNCTIONS
###############################################################################

def handle_live_feed():
    """Handle live feed from IntelRealSenseCamera."""
    st.subheader("📹 Live Video Feed")
    
    recorder = None
    camera = None
    
    # Build the dynamic path from session state
    base_recording_dir = st.session_state["base_dir"]
    rec_subdir = st.session_state["rec_name"]
    recording_dir = os.path.join(base_recording_dir, rec_subdir)

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
        controls_col1, controls_col2, controls_col3, controls_col4 = st.columns(4)
        
        with controls_col1:
            if st.button("🟢 Start Recording", use_container_width=True, 
                        disabled=st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = True
                recorder.start_recording(rec_subdir)  # Use the subdir name
                st.success(f"📝 Recording to: {recorder.get_current_savepath()}")
                global output_path
                output_path = recorder.get_current_savepath()
        
        with controls_col2:
            if st.button("🔴 Stop Recording", use_container_width=True,
                        disabled=not st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = False
                recorder.stop_recording()
                st.info(f"✅ Captured frames")
        
        with controls_col3:
            if st.button("📸 Capture Frame", use_container_width=True):
                wait_message = st.empty()
                wait_message.info("⏳ Waiting for camera to stabilize...")
                time.sleep(1.0)
                
                # Capture frames
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                wait_message.empty()
                
                # Ensure subdirectories exist
                recorder.set_current_savepath(recording_dir)
                os.makedirs(recording_dir, exist_ok=True)
                capture_dir = os.path.join(recording_dir, "captured_frames")
                os.makedirs(capture_dir, exist_ok=True)
                
                # Save the frames
                cv2.imwrite(os.path.join(capture_dir, "image_0.jpg"), color_image)
                np.save(os.path.join(capture_dir, "image_0.npy"), depth_image)
                st.success("✅ Frame captured and saved!")
        
        with controls_col4:
            if st.button("⏹️ Quit", use_container_width=True):
                if recorder and recorder.is_recording:
                    recorder.stop_recording()
                st.session_state["run"] = False
                st.rerun()
        
        # Status indicators
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("Recording Status", 
                      "Active 🟢" if st.session_state.get("recording_status", False) else "")
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
                        rgb_images_dir = os.path.join(recorder.current_savepath, "rgb_images_data_collection")
                        depth_images_dir = os.path.join(recorder.current_savepath, "depth_images_data_collection")
                        os.makedirs(rgb_images_dir, exist_ok=True)
                        os.makedirs(depth_images_dir, exist_ok=True)

                        rgb_path = os.path.join(rgb_images_dir, f"image_{recorder.frame_count}.jpg")
                        cv2.imwrite(rgb_path, color_image)

                        depth_path = os.path.join(depth_images_dir, f"image_{recorder.frame_count}.npy")
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


def handle_timed_recording(duration=10):
    """Record from the camera for a fixed duration."""
    st.subheader("⏲️ Timed Video Recording")
    
    recorder = None
    camera = None

    base_recording_dir = st.session_state["base_dir"]
    rec_subdir = st.session_state["rec_name"]
    recording_dir = os.path.join(base_recording_dir, rec_subdir)
    
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
        
        st.markdown("### 🎮 Controls")
        controls_col1, controls_col2, controls_col3 = st.columns(3)
        
        # Initialize timer state
        if "start_time" not in st.session_state:
            st.session_state.start_time = None
        
        with controls_col1:
            if st.button("🟢 Start Timed Recording", use_container_width=True,
                        disabled=st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = True
                st.session_state.start_time = time.time()
                recorder.start_recording(rec_subdir)
                st.success(f"📝 Recording to: {recorder.get_current_savepath()}")
                global output_path
                output_path = recorder.get_current_savepath()
        
        with controls_col2:
            if st.button("⏹️ Stop Early", use_container_width=True,
                        disabled=not st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = False
                st.session_state.start_time = None
                recorder.stop_recording()
                st.info(f"✅ Recording stopped manually")
                st.session_state["run"] = False
                st.rerun()
        
        with controls_col3:
            if st.button("📸 Capture Frame", use_container_width=True):
                os.makedirs(recording_dir, exist_ok=True)
                wait_message = st.empty()
                wait_message.info("⏳ Waiting for camera to stabilize...")
                time.sleep(1.0)
                
                # Capture frames
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                wait_message.empty()
                
                # Save the frames
                cv2.imwrite(os.path.join(recording_dir, "captured_frame.jpg"), color_image)
                np.save(os.path.join(recording_dir, "captured_frame.npy"), depth_image)
                st.success("✅ Frame captured and saved!")
        
        status_col1, status_col2 = st.columns(2)
        timer_placeholder = st.empty()
        
        with status_col1:
            st.metric("Recording Status", 
                      "Active 🟢" if st.session_state.get("recording_status", False) else "Inactive 🔴")
        with status_col2:
            if st.session_state.get("recording_status", False):
                st.metric("Frames Captured", recorder.frame_count if recorder else 0)
        
        st.markdown("### 📺 Live Preview")
        frame_placeholder = st.empty()
        
        interval = 1/10  # seconds between frames
        last_capture_time = time.time()
        while st.session_state.get("run", True):
            try:
                current_time = time.time()
                
                # Update timer display
                if st.session_state.get("recording_status", False) and st.session_state.start_time is not None:
                    elapsed_time = current_time - st.session_state.start_time
                    remaining_time = max(0, duration - elapsed_time)
                    timer_placeholder.markdown(f"### ⏱️ Time Remaining: {remaining_time:.1f} seconds")
                    
                    if elapsed_time >= duration:
                        st.session_state["recording_status"] = False
                        st.session_state.start_time = None
                        recorder.stop_recording()
                        st.success(f"✅ Recording completed after {duration} seconds!")
                        st.session_state["run"] = False
                        st.rerun()
                        break
                
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                depth_colormap = recorder._normalize_depth_for_display(depth_image)
                display_image = np.hstack((color_image, depth_colormap))
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                
                frame_placeholder.image(display_image, channels="RGB", use_container_width=True,
                                        caption="Live Feed (Color + Depth)")
                
                if (st.session_state.get("recording_status", False) 
                    and (current_time - last_capture_time >= interval) 
                    and recorder.frame_count <= 80):
                    
                    recorder._append_frames(rgb_frame, depth_frame)
                    recorder.frame_count += 1

                    with st.spinner("Saving frames..."):
                        rgb_images_dir = os.path.join(recorder.current_savepath, "rgb_images_data_collection")
                        depth_images_dir = os.path.join(recorder.current_savepath, "depth_images_data_collection")
                        os.makedirs(rgb_images_dir, exist_ok=True)
                        os.makedirs(depth_images_dir, exist_ok=True)

                        rgb_path = os.path.join(rgb_images_dir, f"image_{recorder.frame_count}.jpg")
                        cv2.imwrite(rgb_path, color_image)

                        depth_path = os.path.join(depth_images_dir, f"image_{recorder.frame_count}.npy")
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
    """Handle uploaded video file processing."""
    st.subheader("📤 Upload and Process Video")
    
    uploaded_file = st.file_uploader(
        "Drop your video file here",
        type=["mp4", "avi", "mkv"],
        help="Supported formats: MP4, AVI, MKV"
    )
    
    if not uploaded_file:
        st.info("👆 Please upload a video file to continue")
        return

    with st.spinner("📝 Processing uploaded file..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Convert video in-place for demonstration
        convert_video(video_path, video_path)
    
    st.success("✅ Video processed successfully")
    
    st.markdown("### 🎬 Video Preview")
    st.video(video_path)
    
    st.markdown("### 🔍 Analysis Options")
    col1, col2 = st.columns(2)
    
    with col1:
        analyze_video = st.checkbox(
            "Run video analysis",
            value=True,
            help="Perform detailed analysis of the video content"
        )
        
        if analyze_video:
            if st.button("🚀 Start Analysis", use_container_width=True):
                process_saved_recording(video_path)
    
    with col2:
        if st.button("🎯 Run Inference", use_container_width=True):
            st.session_state["run_inference"] = True
            st.rerun()

###############################################################################
#                    PROCESSING LOGIC AND VIDEO ANALYSIS
###############################################################################

def process_saved_recording(video_path):
    """Perform the entire pipeline (upload, analysis, annotation, RLEF upload, etc.)."""
    st.markdown("### 🔄 Processing Video")
    
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    result_placeholder = st.empty()

    base_recording_dir = st.session_state["base_dir"]
    rec_subdir = st.session_state["rec_name"]
    recording_dir = os.path.join(base_recording_dir, rec_subdir)
    
    try:
        # Initialize progress and payload
        progress_bar = progress_placeholder.progress(0)
        status_placeholder.markdown("⏳ Initializing process...")
        payload_for_cobot_client = {}
        
        # Load payload from a user-configurable path
        payload_path = st.session_state["payload_path"]
        with st.spinner("📋 Loading configuration..."):
            with open(payload_path, "r") as file:
                payload = json.load(file)
            progress_bar.progress(15)
            status_placeholder.markdown("✅ Configuration loaded")

        # Analyze video (upload to GCP + Gemini)
        analyzer = VideoAnalyzer(payload=payload)
        with st.spinner("☁️ Uploading to cloud..."):
            # Example: "test1.mp4" can be replaced or made dynamic
            gcp_url = analyzer.upload_video_to_bucket("test1.mp4", video_path)
            progress_bar.progress(30)
            status_placeholder.markdown("✅ Video uploaded to cloud")
        
        # Get annotations from Gemini
        with st.spinner("🔍 Analyzing video content..."):
            annotations = analyzer.get_gemini_response(gcp_url=gcp_url)
            if annotations:
                with result_placeholder.expander("📊 View Analysis Results", expanded=True):
                    st.json(annotations)
            progress_bar.progress(45)
            status_placeholder.markdown("✅ Analysis complete")
        
        # Upload to RLEF
        rlef_uploader = VideoUploader()
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

        # Object detection + coordinate transformation
        with st.spinner("📍 Processing coordinates and generating predictions..."):
            detector = ObjectDetector(api_key=GEMINI_API_KEY, recording_dir=recording_dir)
            response_coordinates = detector.get_real_world_coordinates(annotations)
            
            if response_coordinates:
                with result_placeholder.expander("🎯 Coordinate Results", expanded=True):
                    st.json(response_coordinates)
            
            progress_bar.progress(75)
            status_placeholder.markdown("✅ Coordinates processed")
            
            # Prepare partial payload for Cobot
            payload_for_cobot_client["fundamental_actions"] = {
                key: {
                    **value,
                    "coordinates": (
                        value["coordinates"].tolist() 
                        if isinstance(value["coordinates"], np.ndarray) 
                        else value["coordinates"]
                    )
                }
                for key, value in response_coordinates.items()
            }
            payload_for_cobot_client["rlef_resource_id"] = rlef_response_text['_id']
            payload_for_cobot_client["video_gcp_url"] = gcp_url
            
            # Attempt to read HAMER predictions
            # Adjust if your actual path is different
            hamer_output_path = os.path.join(recording_dir, "hamer_output", "predictions_hamer.csv")
            try:
                with open(hamer_output_path, "r") as file:
                    csv_hamer_output = file.read()
                payload_for_cobot_client["trajectory_csv"] = csv_hamer_output
                status_placeholder.markdown("✅ HAMER predictions loaded")
            except Exception as e:
                st.warning(f"⚠️ Could not load HAMER predictions: {str(e)}")
                payload_for_cobot_client["trajectory_csv"] = ""
            
            progress_bar.progress(90)

        # Send results to Cobot
        with st.spinner("🤖 Sending data to Cobot..."):
            cobot_client = CobotClient(ip="192.168.0.149", port="8001")
            cobot_client_status = cobot_client.send_trajectory_data(payload_for_cobot_client)
            
            if cobot_client_status:
                status_placeholder.markdown("✅ Data sent to Cobot successfully")
                with result_placeholder.expander("🤖 Cobot Client Payload", expanded=False):
                    st.json(payload_for_cobot_client)
            else:
                st.warning("⚠️ Cobot client response indicates potential issues")
            
        # Update Trajectory in RLEF (upload CSV)
        with st.spinner("🤖 Updating the Trajectory in RLEF..."):
            signed_url = get_signed_url(rlef_response_text['_id'], "predictions_hamer.csv")
            if signed_url:
                upload_hdf5_file(signed_url, hamer_output_path)
                status_placeholder.markdown("✅ Trajectory updated in RLEF")
            else:
                st.warning("⚠️ Failed to update trajectory in RLEF")
            progress_bar.progress(100)
        
        st.success("🎉 All processing steps completed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        progress_placeholder.empty()
        status_placeholder.empty()


def filter_action(input_csv, output_csv):
    """Filter rows (20 to 60 inclusive) from a CSV and save to another CSV."""
    df = pd.read_csv(input_csv)
    filtered_df = df.iloc[20:61]
    filtered_df.to_csv(output_csv, index=False)


def take_images_with_classes_for_inference(
    depth_imagepath=None, 
    rgb_imagepath=None, 
    depth_im=None, 
    rgb_im=None, 
    object_classes=['soda_can', 'glass_cup']
):
    """
    Run object detection inference using Gemini API without hardcoded paths.
    By default, tries to load from session-based 'captured_frames'.
    """
    base_recording_dir = st.session_state["base_dir"]
    rec_subdir = st.session_state["rec_name"]
    recording_dir = os.path.join(base_recording_dir, rec_subdir)
    
    # Provide defaults if not specified
    if depth_imagepath is None:
        depth_imagepath = os.path.join(recording_dir, "captured_frames", "image_0.npy")
    if rgb_imagepath is None:
        rgb_imagepath = os.path.join(recording_dir, "captured_frames", "image_0.jpg")

    try:
        if rgb_im is not None and depth_im is not None:
            # If images were directly passed in memory
            pass
        else:
            # Otherwise, load from filepaths
            st.markdown("### 🔄 Running Object Detection")
            status_placeholder = st.empty()
            result_placeholder = st.empty()
            rgb_im = Image.open(rgb_imagepath)
            depth_im = np.load(depth_imagepath)
        
        with st.spinner("⏳ Getting info from the new scene..."):
            detector = ObjectDetector(api_key=GEMINI_API_KEY, recording_dir=recording_dir)
            object_classes = object_classes if object_classes else ['soda_can', 'white_mug']
            
            with st.spinner("🔍 Detecting object centers..."):
                object_centers = detector.get_object_centers(rgb_im, object_classes[:2])
                if object_centers:
                    object_1_center = object_centers[object_classes[0]][0]
                    object_2_center = object_centers[object_classes[1]][0]
                    status_placeholder.markdown("✅ Object detection complete!")
                    with result_placeholder.expander("📊 Object Detection Results", expanded=True):
                        st.json(object_centers)
                else:
                    st.warning("⚠️ No object centers detected")
                    return
                
            with st.spinner("🔍 Computing Real World Coordinates..."):
                if object_1_center is not None and object_2_center is not None:
                    # Use transform and depth to get real-world coords
                    obj1_rw = transform_coordinates(
                        get_real_world_coordinates(
                            pixel_x=object_1_center[0], 
                            pixel_y=object_1_center[1],
                            image_path=depth_imagepath
                        )
                    )
                    obj2_rw = transform_coordinates(
                        get_real_world_coordinates(
                            pixel_x=object_2_center[0],
                            pixel_y=object_2_center[1],
                            image_path=depth_imagepath
                        )
                    )
                    with result_placeholder.expander("🌍 Real World Coordinates", expanded=False):
                        st.json({
                            object_classes[0]: [float(coord) for coord in obj1_rw],
                            object_classes[1]: [float(coord) for coord in obj2_rw]
                        })
                else:
                    st.warning("⚠️ Object centers are None")
                    return

                # Convert from mm to meters if needed
                obj1_rw_m = [coord / 1000 for coord in obj1_rw]
                obj2_rw_m = [coord / 1000 for coord in obj2_rw]

        # Run model inference to generate trajectory
        with st.spinner("🧠 Generating trajectory from model..."):
            container = [[*obj2_rw, *obj1_rw]]  # Example container
            preds = predict_trajectory(st.session_state["model_path"], container)

            # Save CSV
            csv_filename = f"predicted_trajectory_{int(time.time())}.csv"
            csv_savepath = os.path.join(recording_dir, csv_filename)
            save_predictions_to_csv(preds, csv_savepath)
            filter_action(csv_savepath, csv_savepath)
            with open(csv_savepath, "r") as file:
                csv_contents = file.read()

        # Prepare the payload
        cobot_client_payload = {
            "fundamental_actions": {},
            "trajectory_csv": csv_contents
        }
        frontend_payload = {
            "objects": {},
            "trajectory_csv": csv_contents
        }
        for obj_class, rw_coords in zip(object_classes, [obj1_rw_m, obj2_rw_m]):
            cobot_client_payload['fundamental_actions'][obj_class] = {"coordinates": rw_coords}
            frontend_payload['objects'][obj_class] = {"coordinates": rw_coords}

        # Send to Cobot
        cobot_client = CobotClient(ip="192.168.0.149", port="8001")
        with st.spinner("🤖 Sending data to Cobot..."):
            status = cobot_client.send_trajectory_data(cobot_client_payload)
            if status:
                status_placeholder.markdown("✅ Data sent to Cobot successfully")
                with result_placeholder.expander("🤖 Cobot Client Payload", expanded=False):
                    st.json(frontend_payload)
            else:
                st.warning("⚠️ Cobot client response indicates potential issues")

    except Exception as e:
        st.error(f"❌ Error during object detection: {str(e)}")

###############################################################################
#                               MAIN APP
###############################################################################

def main():
    """Main entry point for the Streamlit application."""
    st.set_page_config(
        page_title="Video Processing Platform",
        page_icon="🎥",
        layout="centered"
    )

    # 1. Initialize session variables
    initialize_session_state()

    # 2. Configure paths in the sidebar (no more hardcoded paths!)
    configure_paths()

    # 3. Create the main sidebar for mode selection
    mode = create_sidebar()

    # 4. Create a header for the main page
    create_header()
    
    # 5. Handle different modes
    if mode == "Live Video Feed":
        handle_live_feed()
    elif mode == "8-Second Recording":
        handle_timed_recording()
    elif mode == "Run Inference":
        take_images_with_classes_for_inference()
    else:  # "Upload Video File"
        handle_uploaded_file()

# Run the app
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.session_state["run"] = False
        st.stop()
