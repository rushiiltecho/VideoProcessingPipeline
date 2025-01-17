from datetime import datetime
import json
import re
from hi_robotics.vision_ai.cameras.intel_realsense_camera import IntelRealSenseCamera
import pandas as pd
import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import time
from PIL import Image

from gemini_constant_api_key import GEMINI_API_KEY
from gemini_oop_object_detection import ObjectDetector, demo_flow
from lit_demo_flow import RealSenseManager
from camera_hi_robotics_realsense_pipeline import RealSenseRecorder
from model.model import predict_trajectory, save_predictions_to_csv
from rlef_video_annotation import VideoUploader
from utils import convert_video, get_real_world_coordinates, get_signed_url, process_images, transform_coordinates, upload_hdf5_file
from vdeo_analysis_ellm_sudio import VideoAnalyzer
from dsr_control_api.dsr_control_api.cobotclient import CobotClient

output_path = None
rec_name = 'Recorded_Demo'
recording_dir = f'recordings/{rec_name}'

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
            ("Live Video Feed", '8-Second Recording',"Upload Video File", "Run Inference"),
            index=2,
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
        controls_col1, controls_col2, controls_col3, controls_col4 = st.columns(4)
        
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
                st.info(f"✅ Captured frames")
        
        with controls_col3:
            if st.button("📸 Capture Frame", use_container_width=True):
                # Show waiting message
                wait_message = st.empty()
                wait_message.info("⏳ Waiting for camera to stabilize...")
                
                # Wait for 1 second
                time.sleep(1.0)
                
                # Capture frames
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Clear the waiting message
                wait_message.empty()
                # print(f"RECORDERSAVEPATH: {recorder.current_savepath}")
                recorder.set_current_savepath('recordings/Recorded_Demo')
                os.makedirs(f'{recorder.current_savepath}', exist_ok=True)
                # Create a 'captured_frames' directory if it doesn't exist
                capture_dir = f'{recorder.current_savepath}/captured_frames'
                # capture_dir = os.path.join(recorder.current_savepath or "recordings/captured_frames")
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


def handle_timed_recording(duration=10):
    """Handle timed recording with automatic stop after specified duration"""
    st.subheader("⏲️ Timed Video Recording")
    
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
        
        # Initialize timer state
        if "start_time" not in st.session_state:
            st.session_state.start_time = None
        
        with controls_col1:
            if st.button("🟢 Start Timed Recording", use_container_width=True, 
                        disabled=st.session_state.get("recording_status", False)):
                st.session_state["recording_status"] = True
                st.session_state.start_time = time.time()
                recorder.start_recording(rec_name)
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
                # Show waiting message
                os.makedirs(f'{recording_dir}', exist_ok=True)
                wait_message = st.empty()
                wait_message.info("⏳ Waiting for camera to stabilize...")
                
                # Wait for 1 second
                time.sleep(1.0)
                
                # Capture frames
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Clear the waiting message
                wait_message.empty()
                
                # Save the frames
                cv2.imwrite(os.path.join(recording_dir, "captured_frame.jpg"), color_image)
                np.save(os.path.join(recording_dir, "captured_frame.npy"), depth_image)
                st.success("✅ Frame captured and saved!")
        
        # Status indicators and timer
        status_col1, status_col2 = st.columns(2)
        timer_placeholder = st.empty()
        
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
                current_time = time.time()
                
                # Update timer display
                if st.session_state.get("recording_status", False) and st.session_state.start_time is not None:
                    elapsed_time = current_time - st.session_state.start_time
                    remaining_time = max(0, duration - elapsed_time)
                    timer_placeholder.markdown(f"### ⏱️ Time Remaining: {remaining_time:.1f} seconds")
                    
                    # Check if recording should stop
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
                
                if st.session_state.get("recording_status", False) and (current_time - last_capture_time >= interval) and recorder.frame_count <= 80:
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
            annotations = analyzer.get_gemini_response(gcp_url=gcp_url)
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
                # csv_hamer_output = process_images(rgb_zip_path, depth_zip_path)
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



def filter_action(input_csv, output_csv):
    # Read CSV with pandas, automatically handling headers
    df = pd.read_csv(input_csv)
    
    # Filter rows (20 to 60 inclusive)
    filtered_df = df.iloc[20:61]
    
    # Save to CSV
    filtered_df.to_csv(output_csv, index=False)

def take_images_with_classes_for_inference(depth_imagepath = 'recordings/Recorded_Demo/captured_frames/image_0.npy', rgb_imagepath = 'recordings/Recorded_Demo/captured_frames/image_0.jpg', depth_im=None, rgb_im=None, object_classes=['soda_can', 'white_mug']):
    """Run object detection inference using Gemini API"""
    try:
        if rgb_im is not None and depth_im is not None:
            rgb_im = rgb_im
            depth_im = depth_im
        
        elif rgb_imagepath and depth_imagepath:
            st.markdown("### 🔄 Running Object Detection")
            status_placeholder = st.empty()
            result_placeholder = st.empty()
            rgb_im = Image.open(rgb_imagepath)
            depth_im = np.load(depth_imagepath)
        else:
            raise ValueError("Either depth_im and rgb_im or depth_imagepath and rgb_imagepath must be provided.")

        with st.spinner("⏳ Getting info from the new scene..."):
            # Initialize the detector
            detector = ObjectDetector(api_key=GEMINI_API_KEY, recording_dir=recording_dir)
            object_classes = object_classes if object_classes else ['soda can', 'white mug']
            print(object_classes)
            # if not annotations or 'objects' not in annotations:
            #     st.error("❌ No objects detected in the video")
            #     return
            with st.spinner("🔍 Detecting object centers..."):
                # Run object center detection
                object_centers = detector.get_object_centers(rgb_im, object_classes[:2])
                if object_centers:
                    object_1_center = object_centers[object_classes[0]][0]
                    object_2_center = object_centers[object_classes[1]][0]
                    print("Object 1 center: ", object_1_center)
                    print("Object 2 center: ", object_2_center)
                    status_placeholder.markdown("✅ Object detection complete!")
                    with result_placeholder.expander("📊 Object Detection Results", expanded=True):
                        st.json(object_centers)
                else:
                    st.warning("⚠️ No object centers detected")
            with st.spinner("🔍 Detecting object centers..."):
                # Get real world coordinates of both the center points:
                if object_1_center is not None and object_2_center is not None and np.any(object_1_center) and np.any(object_2_center):
                    if depth_imagepath:
                        object_1_rw_coords = transform_coordinates(get_real_world_coordinates(pixel_x=object_1_center[0], pixel_y=object_1_center[1], image_path=depth_imagepath))
                        object_2_rw_coords = transform_coordinates(get_real_world_coordinates(pixel_x=object_2_center[0], pixel_y=object_2_center[1], image_path=depth_imagepath))
                    else:
                        object_1_rw_coords = transform_coordinates(get_real_world_coordinates(pixel_x=object_1_center[0], pixel_y=object_1_center[1], im=depth_im))
                        object_2_rw_coords = transform_coordinates(get_real_world_coordinates(pixel_x=object_2_center[0], pixel_y=object_2_center[1], im=depth_im))
                    with result_placeholder.expander("🌍 Real World Coordinates", expanded=False):
                        st.json({
                            object_classes[0]: [float(object_1_rw_coords_i) for object_1_rw_coords_i in object_1_rw_coords],
                            object_classes[1]: [float(object_2_rw_coords_i) for object_2_rw_coords_i in object_2_rw_coords]
                        })
                else:
                    st.warning("⚠️ Object centers are None")
        # PLACEHOLDER for sending the realworld coordinates to the inference and feeding it to the model to generate a trajectory.
        with st.spinner("Getting the inference from model"):
            container = [[*object_1_rw_coords, *object_2_rw_coords]]
            print(f"CONTAINER: {container}")
            preds = predict_trajectory('model/pouring_trajectory_model.pth',container)
            # print(f"PREDS: {preds}")
            savepath = f'{depth_imagepath.split("/")[0]}/{depth_imagepath.split("/")[1]}/predicted_trajectory.csv' if depth_imagepath else f'recordings/predicted_trajectory_{time.time()}'
            # print(f'SAVEPATH {savepath}')
            csv_savedpaths = save_predictions_to_csv(preds,savepath)
            print("CSV SAVED PATHS: ", csv_savedpaths)
            filter_action(savepath, savepath)
            
            with open(savepath, 'r') as file:
                csv_contents = file.read()
                # st.text(csv_contents)

        # Step 2: Send the received generated-trajectory to cobot client
        cobot_client = CobotClient(ip="192.168.0.149", port="8001")

        cobot_client_payload = {
            "fundamental_actions": {}
        }
        frontend_payload = {
            "objects": {}
        }
        for obj_class, rw_coords in zip(object_classes, [object_1_rw_coords, object_2_rw_coords]):
            cobot_client_payload['fundamental_actions'][obj_class] = {"coordinates": rw_coords}
            frontend_payload['objects'][obj_class] = {"coordinates":rw_coords}
        cobot_client_payload['trajectory_csv'] = csv_contents
        frontend_payload['trajectory_csv'] = csv_contents

        print(f"COBOT PAYLOAD: {cobot_client_payload}")

        cobot_client_status = cobot_client.send_trajectory_data(cobot_client_payload)
        # Send to Cobot Client
        with st.spinner("🤖 Sending data to Cobot..."):
            cobot_client = CobotClient(ip="192.168.0.149", port="8001")
            cobot_client_status = cobot_client.send_trajectory_data(cobot_client_payload)
            print(cobot_client_status)
            if cobot_client_status:
                status_placeholder.markdown("✅ Data sent to Cobot successfully")
                with result_placeholder.expander("🤖 Cobot Client Payload", expanded=False):
                    st.json(frontend_payload)
            else:
                st.warning("⚠️ Cobot client response indicates potential issues")

    except Exception as e:
        st.error(f"❌ Error during object detection: {str(e)}")


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
    elif mode == "8-Second Recording":
        handle_timed_recording()
    elif mode == "Run Inference":
        take_images_with_classes_for_inference()
    else:
        handle_uploaded_file()
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.session_state["run"] = False
        st.stop()

