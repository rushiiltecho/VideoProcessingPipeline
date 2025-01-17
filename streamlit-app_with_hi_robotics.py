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
from PIL import Image
import pandas as pd

from gemini_constant_api_key import GEMINI_API_KEY
from gemini_oop_object_detection import ObjectDetector, demo_flow
from lit_demo_flow import RealSenseManager
from camera_hi_robotics_realsense_pipeline import RealSenseRecorder
from model.model import predict_trajectory, save_predictions_to_csv
from rlef_video_annotation import VideoUploader
from utils import convert_video, get_real_world_coordinates, get_signed_url, process_images, transform_coordinates, upload_hdf5_file
from vdeo_analysis_ellm_sudio import VideoAnalyzer
from dsr_control_api.dsr_control_api.cobotclient import CobotClient

# Constants
OUTPUT_DIR = 'recordings'
RECORDING_NAME = 'Recorded_Demo'
RECORDING_DIR = f'{OUTPUT_DIR}/{RECORDING_NAME}'

class SessionState:
    """Class to manage session state variables"""
    def __init__(self):
        if 'initialized' not in st.session_state:
            st.session_state['initialized'] = True
            st.session_state['run'] = True
            st.session_state['recording_status'] = False
            st.session_state['processed_data'] = {}
            st.session_state['analysis_results'] = {}
            st.session_state['cobot_status'] = {}
            st.session_state['current_video'] = None
            st.session_state['object_detection_results'] = {}
            st.session_state['trajectory_data'] = None
            st.session_state['error_log'] = []
            st.session_state['processing_history'] = []

class DataLogger:
    """Class to handle logging and data persistence"""
    @staticmethod
    def log_error(error_message, error_type="ERROR"):
        timestamp = datetime.now().isoformat()
        st.session_state['error_log'].append({
            'timestamp': timestamp,
            'type': error_type,
            'message': str(error_message)
        })

    @staticmethod
    def log_processing(action, status, details=None):
        timestamp = datetime.now().isoformat()
        st.session_state['processing_history'].append({
            'timestamp': timestamp,
            'action': action,
            'status': status,
            'details': details
        })

class VideoProcessor:
    """Class to handle video processing operations"""
    def __init__(self):
        self.camera = None
        self.recorder = None
        self.logger = DataLogger()
        
    def initialize_camera(self):
        """Initialize camera with retry mechanism"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.camera = IntelRealSenseCamera()
                self.recorder = RealSenseRecorder(camera=self.camera)
                self.logger.log_processing("Camera Initialization", "SUCCESS")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    self.logger.log_error(f"Camera initialization failed: {str(e)}")
                    st.error(f"Camera initialization failed after {max_retries} attempts: {str(e)}")
                    return False
    
    def cleanup(self):
        """Cleanup camera resources"""
        try:
            if self.recorder and self.recorder.is_recording:
                self.recorder.stop_recording()
                self.logger.log_processing("Recording Cleanup", "SUCCESS")
            
            if self.camera:
                self.camera.release_camera()
                self.logger.log_processing("Camera Release", "SUCCESS")
        except Exception as e:
            self.logger.log_error(f"Cleanup error: {str(e)}")
            st.error(f"Error during cleanup: {str(e)}")

class UIManager:
    """Class to manage UI components and layout"""
    @staticmethod
    def create_sidebar():
        with st.sidebar:
            st.title("🎥 Video Processing")
            
            st.markdown("### 📝 Navigation")
            mode = st.radio(
                "Select Mode",
                ["Live Feed", "Recording", "Analysis", "Results", "System Status"],
                help="Choose operation mode"
            )
            
            st.markdown("---")
            st.markdown("### ⚙️ Settings")
            if st.checkbox("Show Debug Info", False):
                st.json(st.session_state['error_log'])
                st.json(st.session_state['processing_history'])
            
            if st.button("Reset Session"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()
            
            return mode

    @staticmethod
    def create_header():
        st.markdown("""
        <h1 style='text-align: center;'>
            Advanced Video Processing Platform
        </h1>
        <p style='text-align: center; color: gray;'>
            Record, analyze, and process video data with enhanced capabilities
        </p>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_results_tabs():
        tabs = st.tabs(["Analysis Results", "Object Detection", "Cobot Status", "Trajectory Data", "Processing History"])
        
        with tabs[0]:
            if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
                st.json(st.session_state['analysis_results'])
                if st.button("Export Analysis Results"):
                    UIManager.export_data('analysis_results.json', st.session_state['analysis_results'])
            else:
                st.info("No analysis results available yet")
        
        with tabs[1]:
            if 'object_detection_results' in st.session_state and st.session_state['object_detection_results']:
                st.json(st.session_state['object_detection_results'])
                if st.button("Export Detection Results"):
                    UIManager.export_data('detection_results.json', st.session_state['object_detection_results'])
            else:
                st.info("No object detection results available yet")
        
        with tabs[2]:
            if 'cobot_status' in st.session_state and st.session_state['cobot_status']:
                st.json(st.session_state['cobot_status'])
            else:
                st.info("No cobot status available yet")
        
        with tabs[3]:
            if 'trajectory_data' in st.session_state and st.session_state['trajectory_data'] is not None:
                st.line_chart(pd.DataFrame(st.session_state['trajectory_data']))
                if st.button("Export Trajectory Data"):
                    UIManager.export_data('trajectory.csv', st.session_state['trajectory_data'])
            else:
                st.info("No trajectory data available yet")
        
        with tabs[4]:
            if st.session_state['processing_history']:
                df = pd.DataFrame(st.session_state['processing_history'])
                st.dataframe(df)
            else:
                st.info("No processing history available yet")

    @staticmethod
    def export_data(filename, data):
        """Export data to file"""
        try:
            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(data, f)
            elif filename.endswith('.csv'):
                pd.DataFrame(data).to_csv(filename, index=False)
            st.success(f"Data exported to {filename}")
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")

class VideoRecordingHandler:
    """Class to handle video recording operations"""
    def __init__(self):
        self.processor = VideoProcessor()
        self.logger = DataLogger()
        
    def handle_live_feed(self):
        if not self.processor.initialize_camera():
            return
        
        try:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### 📺 Live Preview")
                frame_placeholder = st.empty()
            
            with col2:
                st.markdown("### 🎮 Controls")
                control_tabs = st.tabs(["Recording", "Capture", "Settings"])
                
                with control_tabs[0]:
                    if st.button("Start Recording", disabled=st.session_state['recording_status']):
                        st.session_state['recording_status'] = True
                        self.processor.recorder.start_recording(RECORDING_NAME)
                        self.logger.log_processing("Recording", "STARTED")
                    
                    if st.button("Stop Recording", disabled=not st.session_state['recording_status']):
                        st.session_state['recording_status'] = False
                        self.processor.recorder.stop_recording()
                        self.logger.log_processing("Recording", "STOPPED")
                        st.success("Recording saved!")
                
                with control_tabs[1]:
                    if st.button("Capture Frame"):
                        self.capture_frame()
                    
                    if st.button("Process Latest Frame"):
                        self.process_latest_frame()
                
                with control_tabs[2]:
                    st.slider("Frame Rate", 1, 30, 10, key="frame_rate")
                    st.checkbox("Show Depth Map", True, key="show_depth")
            
            while st.session_state.get('run', True):
                rgb_frame, depth_frame = self.processor.camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                display_image = self.prepare_display_image(color_image, depth_image)
                frame_placeholder.image(display_image, channels="RGB", use_container_width=True)
                
                if st.session_state['recording_status']:
                    self.save_frame(rgb_frame, depth_frame, color_image, depth_image)
                
                time.sleep(1/st.session_state.get("frame_rate", 10))
                
        except Exception as e:
            self.logger.log_error(f"Live feed error: {str(e)}")
            st.error(f"Error in live feed: {str(e)}")
        finally:
            self.processor.cleanup()

    def prepare_display_image(self, color_image, depth_image):
        try:
            if st.session_state.get("show_depth", True):
                depth_colormap = self.processor.recorder._normalize_depth_for_display(depth_image)
                display_image = np.hstack((color_image, depth_colormap))
            else:
                display_image = color_image
            return cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.logger.log_error(f"Display preparation error: {str(e)}")
            return color_image

    def save_frame(self, rgb_frame, depth_frame, color_image, depth_image):
        try:
            self.processor.recorder._append_frames(rgb_frame, depth_frame)
            frame_count = self.processor.recorder.frame_count
            
            rgb_path = f"{RECORDING_DIR}/rgb_images_data_collection/image_{frame_count}.jpg"
            depth_path = f"{RECORDING_DIR}/depth_images_data_collection/image_{frame_count}.npy"
            
            os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
            os.makedirs(os.path.dirname(depth_path), exist_ok=True)
            
            cv2.imwrite(rgb_path, color_image)
            np.save(depth_path, depth_image)
            
            self.processor.recorder.frame_count += 1
            self.logger.log_processing("Frame Save", "SUCCESS", {'frame_number': frame_count})
        except Exception as e:
            self.logger.log_error(f"Frame save error: {str(e)}")
            st.error(f"Error saving frame: {str(e)}")

    def capture_frame(self):
        try:
            rgb_frame, depth_frame = self.processor.camera.get_frames()
            color_image = np.asanyarray(rgb_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            capture_dir = f'{RECORDING_DIR}/captured_frames'
            os.makedirs(capture_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{capture_dir}/captured_frame_{timestamp}.jpg", color_image)
            np.save(f"{capture_dir}/captured_frame_{timestamp}.npy", depth_image)
            
            st.session_state['current_frame'] = {
                'color': color_image,
                'depth': depth_image,
                'timestamp': timestamp
            }
            
            self.logger.log_processing("Frame Capture", "SUCCESS", {'timestamp': timestamp})
            st.success("Frame captured successfully!")
        except Exception as e:
            self.logger.log_error(f"Frame capture error: {str(e)}")
            st.error(f"Error capturing frame: {str(e)}")

    def process_latest_frame(self):
        if not st.session_state.get('current_frame'):
            st.warning("No frame captured yet!")
            return
        
        try:
            analysis_handler = AnalysisHandler()
            results = analysis_handler.detect_objects(
                st.session_state['current_frame']['color'],
                st.session_state['current_frame']['depth']
            )
            
            if results:
                # Generate trajectory
                container = [[*coord] for coord in results.values()]
                trajectory = predict_trajectory('model/pouring_trajectory_model.pth', container)
                st.session_state['trajectory_data'] = trajectory
                
                # Send to cobot
                cobot_handler = CobotHandler()
                payload = {
                    "fundamental_actions": {
                        class_name: {"coordinates": coords.tolist()}
                        for class_name, coords in results.items()
                    },
                    "trajectory_data": trajectory.tolist()
                }
                cobot_handler.send_to_cobot(payload)
                
                self.logger.log_processing("Frame Processing", "SUCCESS", {
                    'timestamp': st.session_state['current_frame']['timestamp'],
                    'objects_detected': len(results)
                })
        except Exception as e:
            self.logger.log_error(f"Frame processing error: {str(e)}")
            st.error(f"Error processing frame: {str(e)}")

class AnalysisHandler:
    """Class to handle video analysis operations"""
    def __init__(self):
        self.analyzer = VideoAnalyzer(payload={})
        self.detector = ObjectDetector(api_key=GEMINI_API_KEY, recording_dir=RECORDING_DIR)
        self.logger = DataLogger()
    
    def analyze_video(self, video_path):
        try:
            with st.spinner("Analyzing video..."):
                gcp_url = self.analyzer.upload_video_to_bucket("analysis.mp4", video_path)
                self.logger.log_processing("Video Upload", "SUCCESS", {'url': gcp_url})
                
                # Get annotations
                annotations = self.analyzer.get_gemini_response(gcp_url=gcp_url)
                if annotations:
                    st.session_state['analysis_results'] = annotations
                    self.logger.log_processing("Video Analysis", "SUCCESS", {
                        'annotations_count': len(annotations) if isinstance(annotations, list) else 1
                    })
                
                # Process coordinates
                response_coordinates = self.detector.get_real_world_coordinates(annotations)
                if response_coordinates:
                    st.session_state['object_detection_results'] = response_coordinates
                    self.logger.log_processing("Coordinate Processing", "SUCCESS", {
                        'coordinates_count': len(response_coordinates)
                    })
                
                st.success("Analysis completed!")
                return True
        except Exception as e:
            self.logger.log_error(f"Video analysis error: {str(e)}")
            st.error(f"Error in video analysis: {str(e)}")
            return False

    def detect_objects(self, rgb_image, depth_image, object_classes=['soda_can', 'soda_can']):
        try:
            with st.spinner("Detecting objects..."):
                # Get object centers
                object_centers = self.detector.get_object_centers(rgb_image, object_classes)
                if not object_centers:
                    self.logger.log_processing("Object Detection", "WARNING", "No objects detected")
                    st.warning("No objects detected")
                    return
                
                # Process coordinates
                results = {}
                for obj_class, centers in object_centers.items():
                    if centers and centers[0] is not None:
                        coords = transform_coordinates(
                            get_real_world_coordinates(
                                pixel_x=centers[0][0],
                                pixel_y=centers[0][1],
                                im=depth_image
                            )
                        )
                        results[obj_class] = coords
                
                st.session_state['object_detection_results'] = results
                self.logger.log_processing("Object Detection", "SUCCESS", {
                    'objects_detected': len(results)
                })
                return results
        except Exception as e:
            self.logger.log_error(f"Object detection error: {str(e)}")
            st.error(f"Error in object detection: {str(e)}")
            return None

class TrajectoryHandler:
    """Class to handle trajectory generation and processing"""
    def __init__(self):
        self.logger = DataLogger()

    def generate_trajectory(self, coordinates):
        try:
            container = [[*coords] for coords in coordinates.values()]
            trajectory = predict_trajectory('model/pouring_trajectory_model.pth', container)
            
            st.session_state['trajectory_data'] = trajectory
            self.logger.log_processing("Trajectory Generation", "SUCCESS", {
                'points_count': len(trajectory)
            })
            
            # Save trajectory to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"{RECORDING_DIR}/trajectories/trajectory_{timestamp}.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            save_predictions_to_csv(trajectory, csv_path)
            
            return trajectory
        except Exception as e:
            self.logger.log_error(f"Trajectory generation error: {str(e)}")
            st.error(f"Error generating trajectory: {str(e)}")
            return None

class CobotHandler:
    """Class to handle cobot operations"""
    def __init__(self):
        self.client = CobotClient(ip="192.168.0.149", port="8001")
        self.logger = DataLogger()
    
    def send_to_cobot(self, payload):
        try:
            with st.spinner("Sending data to cobot..."):
                status = self.client.send_trajectory_data(payload)
                if status:
                    st.session_state['cobot_status'] = {
                        'status': 'success',
                        'timestamp': datetime.now().isoformat(),
                        'payload': payload
                    }
                    self.logger.log_processing("Cobot Communication", "SUCCESS", {
                        'payload_size': len(str(payload))
                    })
                    st.success("Data sent to cobot successfully!")
                    return True
                else:
                    self.logger.log_processing("Cobot Communication", "ERROR", "Failed to send data")
                    st.error("Failed to send data to cobot")
                    return False
        except Exception as e:
            self.logger.log_error(f"Cobot communication error: {str(e)}")
            st.error(f"Error communicating with cobot: {str(e)}")
            return False

class SystemStatusHandler:
    """Class to handle system status and diagnostics"""
    def __init__(self):
        self.logger = DataLogger()

    def show_system_status(self):
        st.markdown("### 🔧 System Status Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Processing Statistics")
            if st.session_state['processing_history']:
                df = pd.DataFrame(st.session_state['processing_history'])
                success_rate = (df['status'] == 'SUCCESS').mean() * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Group by action and status
                action_stats = df.groupby(['action', 'status']).size().unstack(fill_value=0)
                st.bar_chart(action_stats)
        
        with col2:
            st.markdown("#### ⚠️ Error Log")
            if st.session_state['error_log']:
                error_df = pd.DataFrame(st.session_state['error_log'])
                st.dataframe(error_df)
            else:
                st.info("No errors logged")
        
        st.markdown("#### 💾 Storage Status")
        if os.path.exists(RECORDING_DIR):
            total_size = 0
            file_count = 0
            for dirpath, dirnames, filenames in os.walk(RECORDING_DIR):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
                    file_count += 1
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Total Files", file_count)
            with col4:
                st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")

def main():
    st.set_page_config(
        page_title="Advanced Video Processing",
        page_icon="🎥",
        layout="wide"
    )
    
    # Initialize session state
    SessionState()
    
    # Create UI components
    ui = UIManager()
    mode = ui.create_sidebar()
    ui.create_header()
    
    # Handle different modes
    if mode == "Live Feed":
        recording_handler = VideoRecordingHandler()
        recording_handler.handle_live_feed()
    
    elif mode == "Recording":
        st.markdown("### 📹 Video Recording")
        tabs = st.tabs(["New Recording", "Upload Video", "Recording History"])
        
        with tabs[0]:
            recording_handler = VideoRecordingHandler()
            recording_handler.handle_live_feed()
        
        with tabs[1]:
            uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
            if uploaded_file:
                st.session_state['current_video'] = uploaded_file
                st.video(uploaded_file)
                
                if st.button("Process Uploaded Video"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        analysis_handler = AnalysisHandler()
                        analysis_handler.analyze_video(tmp_file.name)
        
        with tabs[2]:
            if os.path.exists(RECORDING_DIR):
                recordings = []
                for root, dirs, files in os.walk(RECORDING_DIR):
                    for file in files:
                        if file.endswith(('.mp4', '.avi', '.mov')):
                            recordings.append(os.path.join(root, file))
                
                if recordings:
                    selected_recording = st.selectbox("Select Recording", recordings)
                    if selected_recording:
                        st.video(selected_recording)
                        if st.button("Analyze Selected Recording"):
                            analysis_handler = AnalysisHandler()
                            analysis_handler.analyze_video(selected_recording)
                else:
                    st.info("No recordings found")
    
    elif mode == "Analysis":
        st.markdown("### 🔍 Analysis")
        tabs = st.tabs(["Video Analysis", "Object Detection", "Trajectory Generation"])
        
        with tabs[0]:
            if st.session_state.get('current_video'):
                if st.button("Analyze Current Video"):
                    analysis_handler = AnalysisHandler()
                    analysis_handler.analyze_video(st.session_state['current_video'])
            else:
                st.info("Please upload or record a video first")
        
        with tabs[1]:
            if st.session_state.get('current_frame'):
                st.image(st.session_state['current_frame']['color'], caption="Current Frame")
                if st.button("Detect Objects"):
                    analysis_handler = AnalysisHandler()
                    results = analysis_handler.detect_objects(
                        st.session_state['current_frame']['color'],
                        st.session_state['current_frame']['depth']
                    )
                    if results:
                        trajectory_handler = TrajectoryHandler()
                        trajectory = trajectory_handler.generate_trajectory(results)
                        
                        if trajectory is not None:
                            cobot_handler = CobotHandler()
                            payload = {
                                "fundamental_actions": {
                                    class_name: {"coordinates": coords.tolist()}
                                    for class_name, coords in results.items()
                                },
                                "trajectory_data": trajectory.tolist()
                            }
                            cobot_handler.send_to_cobot(payload)
            else:
                st.info("Please capture a frame first")
        
        with tabs[2]:
            if st.session_state.get('object_detection_results'):
                if st.button("Generate New Trajectory"):
                    trajectory_handler = TrajectoryHandler()
                    trajectory_handler.generate_trajectory(st.session_state['object_detection_results'])
            else:
                st.info("Please perform object detection first")
    
    elif mode == "Results":
        st.markdown("### 📊 Results Dashboard")
        ui.show_results_tabs()
    
    elif mode == "System Status":
        system_status = SystemStatusHandler()
        system_status.show_system_status()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.session_state["run"] = False
        st.stop()
    except Exception as e:
        DataLogger.log_error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")
        st.stop()