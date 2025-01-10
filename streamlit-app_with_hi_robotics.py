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
from utils import convert_video, process_images
from vdeo_analysis_ellm_sudio import VideoAnalyzer
from dsr_control_api.dsr_control_api.cobotclient import CobotClient

output_path = None

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
    """Handles live feed using RealSenseRecorder with improved device handling"""
    st.subheader("Live Video Feed")
    
    # Initialize RealSense recorder
    recorder = None
    camera = None
    
    try:
        # Initialize with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                camera = IntelRealSenseCamera()
                recorder = RealSenseRecorder(camera=camera)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                else:
                    raise e
        
        if not recorder:
            st.error("Failed to initialize camera after multiple attempts")
            return
            
        # UI Controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Start Recording"):
                if not recorder.is_recording:
                    recorder.start_recording()
                    st.success(f"Recording started. Output directory: {recorder.get_current_savepath()}")
                    global output_path
                    output_path = recorder.get_current_savepath()
        
        with col2:
            if st.button("Stop Recording"):
                if recorder.is_recording:
                    recorder.stop_recording()
                    st.info(f"Recording stopped. Frames captured: {recorder.frame_count}")
        
        with col3:
            if st.button("Quit"):
                if recorder and recorder.is_recording:
                    recorder.stop_recording()
                st.session_state["run"] = False
                st.rerun()
        
        # Display frames
        frame_placeholder = st.empty()
        
        while st.session_state.get("run", True):
            try:
                # Wait for frameset with timeout
                rgb_frame, depth_frame = camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Create depth colormap for display
                depth_colormap = recorder._normalize_depth_for_display(depth_image)
                
                # Stack color and depth horizontally
                display_image = np.hstack((color_image, depth_colormap))
                
                # Convert BGR to RGB for Streamlit
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(display_image, channels="RGB", use_container_width=True)
                
                # If recording, save frames
                if recorder.is_recording:
                    prev_time = time.time()
                    recorder._append_frames(rgb_frame, depth_frame)
                    recorder.frame_count += 1

                    current_time = time.time()
                    if current_time - prev_time < 1.0 / recorder.fps:
                        # Save RGB image:
                        rgb_path = f"{recorder.current_savepath}/rgb_images_data_collection/image_{recorder.frame_count}.jpg"
                        cv2.imwrite(rgb_path, color_image)

                        # Save depth image as .npy:
                        depth_path = f"{recorder.current_savepath}/depth_images_data_collection/image_{recorder.frame_count}.npy"
                        np.save(depth_path, depth_image)

                        # Increment image counter and update last capture time
                        print(f"Saved : {recorder.frame_count}")
                        prev_time = current_time
                    
            except Exception as e:
                st.error(f"Error during frame capture: {str(e)}")
                break
                
    except Exception as e:
        st.error(f"Failed to initialize RealSense camera: {str(e)}")
        return
        
    finally:
        # Cleanup
        if recorder and recorder.is_recording:
            try:
                recorder.stop_recording()
            except Exception as e:
                st.error(f"Error during cleanup: {str(e)}")
        
        if camera:
            try:
                camera.release_camera()
            except Exception as e:
                st.error(f"Error releasing camera: {str(e)}")
                
        st.session_state["run"] = False

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
    original_path = f"{os.path.abspath(uploaded_file.name)}/recordings"
    upload_directory = os.path.dirname(original_path)
    video_path = tfile.name
    st.write(f"Original file <{uploaded_file.name}> path: {original_path} in {upload_directory}")
    convert_video(video_path, video_path)
    print(f"VIDEO PATH ==================== {video_path}")

    st.success(f"Video uploaded successfully")
    st.video(video_path)

    # Option to analyze
    if st.checkbox("Analyze this video now?", value=True):
        process_saved_recording(video_path)

def __process_saved_recording(video_path):
    """Process a recording through video analysis and RLEF upload"""
    progress_bar = st.progress(0)
    
    try:
        # Step 1: Load payload
        payload_for_cobot_client = {}
        st.write("Loading payload...")
        with open("payload.json", "r") as file:
            payload = json.load(file)
        progress_bar.progress(20)

        # Step 2: Upload to GCP and analyze
        st.write("Uploading to GCP and analyzing...")
        analyzer = VideoAnalyzer(payload=payload)
        gcp_url = analyzer.upload_video_to_bucket(
            f"test1.mp4",
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
        progress_bar.progress(50)


        # Step 4: Upload to RLEF
        with st.spinner("Uploading to RLEF..."):
            rlef_uploader = VideoUploader()
            #TODO: change the arguments in upload to rlef to include the updated payload alongwith the prediction CSV.
            status, rlef_response_text = rlef_uploader.upload_to_rlef(
                rlef_url="https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/",
                video_filepath=video_path,
                video_annotations=annotations,
                csv_filepath=None
            )
        
            
            if status == 200:
                st.success("Processing completed successfully!")
            else:
                st.warning(f"RLEF upload returned status code: {status}")

        response_coordinates= None
        coordinates = None
        boxes = None
        csv_hamer_output = None

        with st.spinner("Generating response..."):
            try:
                recording_dir = 'recordings/Recorded_Demo_1'
                rgb_zip_path = f'{recording_dir}/archives/rgb_images_data_collection.zip'
                depth_zip_path = f'{recording_dir}/archives/depth_images_data_collection.zip'
                st.write(f'Getting Coordinates from the video Analysis: ')
                # response_coordinates = demo_flow(recording_dir=recording_dir, response_annotations=annotations)
                detector = ObjectDetector(api_key=GEMINI_API_KEY, recording_dir= recording_dir)
                response_coordinates = detector.get_real_world_coordinates(annotations)
                boxes = [response_coordinates[i]['box'] for i in response_coordinates.keys()]
                coordinates = [response_coordinates[i]['coordinates'] for i in response_coordinates.keys()]
                cobot_client = CobotClient(ip="192.168.0.149", port="8001")
                print("RESPONSE COORDINATES TO SEND: ", response_coordinates)
                # ========================================================================================
                # for key, value in response_coordinates.items():
                #     if re.search(r'picking up', key, re.IGNORECASE):
                #         task_type = "pick_object"
                #     elif re.search(r'placing', key, re.IGNORECASE):
                #         task_type = "place_object"
                #     else:
                #         continue
                #     # TODO: Send a different payload to this with the updated flow.
                    # res = cobot_client.send_task(task_type=task_type, task_data=value['coordinates'])
                    # print("COBOT_API_RESPONSE: ", res)
                # ========================================================================================
                # payload_for_cobot_client["fundamental_actions"] = response_coordinates  
                payload_for_cobot_client["fundamental_actions"] = {
                    key: {
                        **value,
                        "coordinates": value["coordinates"].tolist() if isinstance(value["coordinates"], np.ndarray) else value["coordinates"]
                    }
                    for key, value in response_coordinates.items()
                }  
                payload_for_cobot_client["rlef_resource_id"] = rlef_response_text['_id']
                payload_for_cobot_client["video_gcp_url"] = gcp_url
                # --------------------------------------
                # csv_hamer_output = process_images(rgb_zip_path=rgb_zip_path, depth_zip_path=depth_zip_path, output_dir=f"{recording_dir}/hamer_output")
                with open(f"{recording_dir}/hamer_output/predictions_hamer_sample.csv", "r") as file:
                    csv_hamer_output = file.read()
                # --------------------------------------
                # rlef_uploader.process_and_upload_csv(video_bucket_id=payload_for_cobot_client["rlef_resource_id"], csv_filepath=f'{recording_dir}/hamer_output/predictions_hamer_sample.csv', csv_filename='predictions_hamer_sample.csv')
                payload_for_cobot_client["trajectory_csv"] = csv_hamer_output if csv_hamer_output else ""
                # ============ PLACEHOLDER: send the data to cobot client ================
                print("==================== PAYLOAD FOR COBOT CLIENT: ====================\n", payload_for_cobot_client)
                # ========================================================================
                if response_coordinates:
                    st.write("Coordinate Location Received...:")
                    st.json(response_coordinates)
                    print(type(payload_for_cobot_client))
                cobot_client_status = cobot_client.send_trajectory_data(payload_for_cobot_client)
                print(f"COBOT_CLIENT_STATUS: Updated Trajectory CSV {cobot_client_status}")
            except Exception as e:
                st.error(f"Error analyzing video: {e}")
                return

        
        progress_bar.progress(100)

            
    except Exception as e:
        st.error(f"Error processing recording: {str(e)}")
        progress_bar.progress(100)


def process_saved_recording(video_path):
    """Enhanced processing with better tracking and organized output display"""
    st.markdown("### 🔄 Processing Video")
    
    # Create containers for different sections
    progress_container = st.container()
    tabs_container = st.container()
    
    with progress_container:
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        status_placeholder.markdown("⏳ Initializing process...")
    
    # Create tabs for different outputs
    with tabs_container:
        analysis_tab, coordinates_tab, rlef_tab, cobot_tab, logs_tab = st.tabs([
            "📊 Analysis Results", 
            "📍 Coordinates", 
            "🔄 RLEF Status", 
            "🤖 Cobot Data",
            "📝 Process Logs"
        ])
    
    try:
        payload_for_cobot_client = {}
        
        # Load payload
        with st.spinner("📋 Loading configuration..."):
            with open("payload.json", "r") as file:
                payload = json.load(file)
            progress_bar.progress(15)
            status_placeholder.markdown("✅ Configuration loaded")
            with logs_tab:
                st.success("Configuration loaded successfully")
                st.json(payload)
        
        # Upload and analyze
        with st.spinner("☁️ Uploading to cloud..."):
            analyzer = VideoAnalyzer(payload=payload)
            gcp_url = analyzer.upload_video_to_bucket("test1.mp4", video_path)
            progress_bar.progress(30)
            status_placeholder.markdown("✅ Video uploaded to cloud")
            with logs_tab:
                st.success("Video uploaded to GCP")
                st.code(gcp_url, language="text")
        
        # Get annotations
        with st.spinner("🔍 Analyzing video content..."):
            annotations = analyzer.get_ellm_response()
            if annotations:
                with analysis_tab:
                    st.success("✅ Video Analysis Complete")
                    st.json(annotations)
                with logs_tab:
                    st.success("Analysis completed successfully")
            progress_bar.progress(45)
            status_placeholder.markdown("✅ Analysis complete")
        
        # Upload to RLEF
        with st.spinner("📤 Uploading results to RLEF..."):
            rlef_uploader = VideoUploader()
            status, rlef_response_text = rlef_uploader.upload_to_rlef(
                rlef_url="https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/",
                video_filepath=video_path,
                video_annotations=annotations,
                csv_filepath=None
            )
            progress_bar.progress(60)
            status_placeholder.markdown("✅ Results uploaded to RLEF")
            
            with rlef_tab:
                if status == 200:
                    st.success("✅ Upload to RLEF Successful")
                    st.json(rlef_response_text)
                else:
                    st.warning(f"⚠️ RLEF Upload Status: {status}")
            with logs_tab:
                st.info(f"RLEF upload completed with status: {status}")
        
        # Process coordinates and HAMER predictions
        with st.spinner("📍 Processing coordinates and generating predictions..."):
            recording_dir = 'recordings/Recorded_Demo_1'
            rgb_zip_path = f'{recording_dir}/archives/rgb_images_data_collection.zip'
            depth_zip_path = f'{recording_dir}/archives/depth_images_data_collection.zip'
            
            # Get coordinates
            detector = ObjectDetector(api_key=GEMINI_API_KEY, recording_dir=recording_dir)
            response_coordinates = detector.get_real_world_coordinates(annotations)
            
            if response_coordinates:
                with coordinates_tab:
                    st.success("✅ Coordinates Extracted Successfully")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### 📍 Coordinate Data")
                        st.json(response_coordinates)
                    with col2:
                        st.markdown("### 📊 Visualization")
                        boxes = [response_coordinates[i]['box'] for i in response_coordinates.keys()]
                        coordinates = [response_coordinates[i]['coordinates'] for i in response_coordinates.keys()]
                        # You could add a visualization here if needed
                
                with logs_tab:
                    st.success("Coordinates processing completed")
                    st.write("Boxes:", boxes)
                    st.write("Coordinates:", coordinates)
            
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
                with open(f"{recording_dir}/hamer_output/predictions_hamer_sample.csv", "r") as file:
                    csv_hamer_output = file.read()
                payload_for_cobot_client["trajectory_csv"] = csv_hamer_output
                status_placeholder.markdown("✅ HAMER predictions loaded")
                
                with cobot_tab:
                    st.success("✅ HAMER Predictions Loaded")
                    with st.expander("View HAMER Predictions"):
                        st.code(csv_hamer_output, language="csv")
                
                with logs_tab:
                    st.success("HAMER predictions loaded successfully")
            except Exception as e:
                error_msg = f"⚠️ Could not load HAMER predictions: {str(e)}"
                with cobot_tab:
                    st.warning(error_msg)
                with logs_tab:
                    st.error(error_msg)
                payload_for_cobot_client["trajectory_csv"] = ""
            
            progress_bar.progress(90)
        
        # Send to Cobot Client
        with st.spinner("🤖 Sending data to Cobot..."):
            cobot_client = CobotClient(ip="192.168.0.149", port="8001")
            cobot_client_status = cobot_client.send_trajectory_data(payload_for_cobot_client)
            
            with cobot_tab:
                st.markdown("### 🤖 Cobot Client Data")
                if cobot_client_status:
                    st.success("✅ Data sent to Cobot successfully")
                    st.json(payload_for_cobot_client)
                else:
                    st.warning("⚠️ Cobot client response indicates potential issues")
                
                # Display cobot client configuration
                with st.expander("Cobot Client Configuration"):
                    st.write("IP Address:", "192.168.0.149")
                    st.write("Port:", "8001")
                    st.write("Status:", cobot_client_status)
            
            with logs_tab:
                st.info(f"Cobot client status: {cobot_client_status}")
                st.json({"payload_size": len(str(payload_for_cobot_client))})
            
            progress_bar.progress(100)
            status_placeholder.markdown("✅ All steps completed")
        
        # Final success message
        st.success("🎉 All processing steps completed successfully!")
        
        # Summary in logs tab
        with logs_tab:
            st.markdown("### 📋 Process Summary")
            st.write("1. ✅ Configuration loaded")
            st.write("2. ✅ Video uploaded to GCP")
            st.write("3. ✅ Video analysis completed")
            st.write("4. ✅ RLEF upload finished")
            st.write("5. ✅ Coordinates processed")
            st.write("6. ✅ HAMER predictions loaded")
            st.write("7. ✅ Cobot client data sent")
        
    except Exception as e:
        error_msg = f"❌ Error during processing: {str(e)}"
        st.error(error_msg)
        with logs_tab:
            st.error(error_msg)
            st.error("Stack trace:", exception=True)
        progress_placeholder.empty()
        status_placeholder.empty()


# Use session state to help with stop/clean mechanism
if "run" not in st.session_state:
    st.session_state["run"] = True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.session_state["run"] = False
        st.stop()
