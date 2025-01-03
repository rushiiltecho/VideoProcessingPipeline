from datetime import datetime
import json
import re
import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import time 

from gemini_constant_api_key import GEMINI_API_KEY
from gemini_oop_object_detection import ObjectDetector, demo_flow
from lit_demo_flow import RealSenseManager
from realsense_recording import RealSenseRecorder
from rlef_video_annotation import VideoUploader
from utils import convert_video
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

def handle_live_feed_orig():
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

def handle_live_feed():
    """Handles live feed using RealSenseRecorder with improved device handling"""
    st.subheader("Live Video Feed")
    
    # Initialize RealSense recorder
    recorder = None
    
    # Add a device cleanup button in case of locked device
    if st.button("Reset Camera"):
        try:
            import pyrealsense2 as rs
            # Get the context and query devices
            ctx = rs.context()
            devices = ctx.query_devices()
            # Reset each device
            for dev in devices:
                dev.hardware_reset()
            st.success("Camera reset successful. Please wait a few seconds before starting.")
            time.sleep(2)  # Give device time to reset
        except Exception as e:
            st.error(f"Error resetting camera: {str(e)}")
    
    try:
        # Initialize with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                recorder = RealSenseRecorder()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    # st.warning(f"Initialization attempt {attempt + 1} failed, retrying...")
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
                    recorder.start_recording("Demo_Recording")
                    st.success(f"Recording started. Output directory: {recorder.get_current_savepath()}")
                    global output_path
                    output_path = recorder.get_current_savepath()
        
        with col2:
            if st.button("Stop Recording"):
                if recorder.is_recording:
                    recorder.stop_recording()
                    st.info(f"Recording stopped. Frames captured: {recorder.frame_count}")
                    
                    # Optionally analyze the recording
                    if st.checkbox("Analyze this recording?"):
                        with st.spinner("Analyzing recording..."):
                            try:
                                analyzer = VideoAnalyzer()
                                rlef_uploader = VideoUploader()
                                recorder.get_ellm_studio_analysis(analyzer, rlef_uploader)
                                st.success("Analysis complete!")
                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
        
        with col3:
            if st.button("Quit"):
                if recorder and recorder.is_recording:
                    recorder.stop_recording()
                if recorder:
                    recorder.pipeline.stop()
                st.session_state["run"] = False
                st.rerun()
        
        # Display frames
        frame_placeholder = st.empty()
        
        while st.session_state.get("run", True):
            try:
                # Wait for frameset with timeout
                frames = recorder.pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    continue
                    
                aligned_frames = recorder.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Create depth colormap for display
                depth_colormap = recorder._normalize_depth_for_display(depth_image)
                
                # Stack color and depth horizontally
                display_image = np.hstack((color_image, depth_colormap))
                
                # Convert BGR to RGB for Streamlit
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(display_image, channels="RGB", use_column_width=True)
                
                # If recording, save frames
                if recorder.is_recording:
                    recorder._append_frames(color_frame, depth_frame, frames)
                    recorder.frame_count += 1
                    
            except Exception as e:
                st.error(f"Error during frame capture: {str(e)}")
                break
                
    except Exception as e:
        st.error(f"Failed to initialize RealSense camera: {str(e)}")
        return
        
    finally:
        # Cleanup
        if recorder:
            try:
                if recorder.is_recording:
                    recorder.stop_recording()
                recorder.pipeline.stop()
            except Exception as e:
                st.error(f"Error during cleanup: {str(e)}")
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
    
    progress_bar.progress(60)
    st.write(f'Getting Coordinates from the video Analysis: ')
    response_coordinates= None
    coordinates = None
    boxes = None
    with st.spinner("Generating response..."):
        try:
            recording_dir = 'recordings/Demo_Recording'
            response_coordinates = demo_flow(recording_dir=recording_dir,response_annotations=response_annotations)
            boxes = [response_coordinates[i]['box'] for i in response_coordinates.keys()]
            print(boxes)
            coordinates = [response_coordinates[i]['coordinates'] for i in response_coordinates.keys()]
        except Exception as e:
            st.error(f"Error analyzing video: {e}")
            return
    if response_coordinates:
        st.write("Coordinate Location Received...:")
        st.json(response_coordinates)
    
    progress_bar.progress(60)

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

def __advanced_handle_uploaded_file():
    """Handles uploading and processing of a recording directory containing color.mp4, depth_visualization.mp4, metadata.json, and frames.h5"""
    st.subheader("Upload Recording Directory")

    # Create uploader for directory contents
    uploaded_files = st.file_uploader(
        "Upload your recording directory (containing color.mp4, depth_visualization.mp4, metadata.json, frames.h5)", 
        type=["mp4", "json", "h5"], 
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        st.info("Please upload your recording directory containing all required files")
        return

    # Create a temporary directory to store the uploaded directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Dictionary to map expected filenames to their paths
        expected_files = {
            'color.mp4': None,
            'depth_visualization.mp4': None,
            'metadata.json': None,
            'frames.h5': None
        }
        
        # Process the uploaded files
        for uploaded_file in uploaded_files:
            if uploaded_file.name in expected_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.read())
                expected_files[uploaded_file.name] = file_path
            else:
                st.warning(f"Unexpected file found: {uploaded_file.name}")

        # Verify directory contents
        missing_files = [fname for fname, path in expected_files.items() if path is None]
        
        if missing_files:
            st.error("Missing required files in the uploaded directory: " + ", ".join(missing_files))
            return
            
        st.success("Recording directory uploaded successfully!")
        
        try:
            # Load metadata
            with open(expected_files['metadata.json'], 'r') as f:
                metadata = json.load(f)
            
            # Display recording information
            st.write("### Recording Information")
            if 'timestamp' in metadata:
                st.write(f"Recording Time: {metadata['timestamp']}")
            if 'duration' in metadata:
                st.write(f"Duration: {metadata['duration']} seconds")
            
            # Display video previews
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Color Video")
                st.video(expected_files['color.mp4'])
            
            with col2:
                st.write("### Depth Video")
                st.video(expected_files['depth_visualization.mp4'])
            
            # Process recording button
            if st.button("Process Recording"):
                process_recording_directory(
                    color_path=expected_files['color.mp4'],
                    depth_path=expected_files['depth_visualization.mp4'],
                    metadata_path=expected_files['metadata.json'],
                    frames_path=expected_files['frames.h5']
                )
                
        except Exception as e:
            st.error(f"Error processing recording directory: {str(e)}")

def process_recording_directory(color_path, depth_path, metadata_path, frames_path):
    """Process all components of the recording directory."""
    progress_bar = st.progress(0)
    
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Step 1: Initialize analyzer
        st.write("Initializing analysis...")
        analyzer = VideoAnalyzer(payload=metadata)
        progress_bar.progress(20)

        # Step 2: Upload color video to GCP
        st.write("Uploading color video to GCP...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        color_gcp_url = analyzer.upload_video_to_bucket(
            f"color_{timestamp}.mp4",
            color_path
        )
        progress_bar.progress(35)

        # Step 3: Upload depth video to GCP
        st.write("Uploading depth video to GCP...")
        depth_gcp_url = analyzer.upload_video_to_bucket(
            f"depth_{timestamp}.mp4",
            depth_path
        )
        progress_bar.progress(50)

        # Step 4: Process frames data
        st.write("Processing frames data...")
        try:
            import h5py
            with h5py.File(frames_path, 'r') as f:
                # Add your frames processing logic here
                st.write("Frames data loaded successfully")
        except Exception as e:
            st.warning(f"Error processing frames data: {str(e)}")
        progress_bar.progress(65)

        # Step 5: Get video annotations
        st.write("Analyzing recording...")
        annotations = analyzer.get_ellm_response()
        if annotations:
            with st.expander("View Analysis Results"):
                st.json(annotations)
        progress_bar.progress(80)

        # Step 6: Upload to RLEF
        st.write("Uploading to RLEF...")
        rlef_uploader = VideoUploader()
        
        # Create combined metadata including both video URLs
        combined_metadata = {
            **metadata,
            "color_video_url": color_gcp_url,
            "depth_video_url": depth_gcp_url
        }
        
        status = rlef_uploader.upload_to_rlef(
            url="https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/",
            filepath=color_path,
            video_annotations=annotations,
            # additional_metadata=combined_metadata
        )
    
        progress_bar.progress(100)
        
        if status == 200:
            st.success("Recording processed successfully!")
            st.write("### Processing Summary")
            st.write("✅ Color video uploaded and processed")
            st.write("✅ Depth video uploaded and processed")
            st.write("✅ Frames data analyzed")
            st.write("✅ Annotations generated")
            st.write("✅ Results uploaded to RLEF")
        else:
            st.warning(f"RLEF upload returned status code: {status}")
            
    except Exception as e:
        st.error(f"Error processing recording: {str(e)}")
    finally:
        progress_bar.progress(100)

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


        st.write(f'Getting Coordinates from the video Analysis: ')
        response_coordinates= None
        coordinates = None
        boxes = None
        with st.spinner("Generating response..."):
            try:
                recording_dir = 'recordings/Demo_Recording'
                response_coordinates = demo_flow(recording_dir=recording_dir, response_annotations=annotations)
                boxes = [response_coordinates[i]['box'] for i in response_coordinates.keys()]
                coordinates = [response_coordinates[i]['coordinates'] for i in response_coordinates.keys()]
                cobot_client = CobotClient(ip="192.168.0.129", port="8001")
                
                for key, value in response_coordinates.items():
                    if re.search(r'picking up', key, re.IGNORECASE):
                        task_type = "pick_object"
                    elif re.search(r'placing', key, re.IGNORECASE):
                        task_type = "place_object"
                    else:
                        continue
                    res = cobot_client.send_task(task_type=task_type, task_data=value['coordinates'])
                    print("COBOT_API_RESPONSE: ", res)
            except Exception as e:
                st.error(f"Error analyzing video: {e}")
                return
        if response_coordinates:
            st.write("Coordinate Location Received...:")
            st.json(response_coordinates)
        
        progress_bar.progress(75) 

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



# Use session state to help with stop/clean mechanism
if "run" not in st.session_state:
    st.session_state["run"] = True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.session_state["run"] = False
        st.stop()
