from rlef_video_annotation import VideoUploader
from utils import convert_video
from vdeo_analysis_ellm_sudio import VideoAnalyzer
import pyrealsense2 as rs
import cv2
import numpy as np
import h5py
import json
from datetime import datetime
import os

class RealSenseRecorder:
    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.width = 640
        self.height = 480
        self.fps = 30
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        
        # Get depth scale
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        
        # Get camera intrinsics
        self.color_profile = self.profile.get_stream(rs.stream.color)
        self.depth_profile = self.profile.get_stream(rs.stream.depth)
        self.color_intrinsics = self.color_profile.as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = self.depth_profile.as_video_stream_profile().get_intrinsics()
        
        # Recording state
        self.recording_id = None
        self.frame_count = 0
        self.is_recording = False
        self.h5_file = None
        self.video_writer = None
        self.current_savepath = None
        self.depth_video_writer = None  # For visualization of depth data
        self.recording_stopped_callback = None

    def set_recording_stopped_callback(self, callback):
        """
        Register a callback to be invoked each time a recording stops.
        """
        self.recording_stopped_callback = callback

    def _create_intrinsics_dict(self):
        """Create a dictionary of camera intrinsics"""
        return {
            "color_intrinsics": {
                "fx": self.color_intrinsics.fx,
                "fy": self.color_intrinsics.fy,
                "ppx": self.color_intrinsics.ppx,
                "ppy": self.color_intrinsics.ppy,
                "model": str(self.color_intrinsics.model),
                "coeffs": self.color_intrinsics.coeffs
            },
            "depth_intrinsics": {
                "fx": self.depth_intrinsics.fx,
                "fy": self.depth_intrinsics.fy,
                "ppx": self.depth_intrinsics.ppx,
                "ppy": self.depth_intrinsics.ppy,
                "model": str(self.depth_intrinsics.model),
                "coeffs": self.depth_intrinsics.coeffs
            },
            "depth_scale": self.depth_scale
        }

    def start_recording(self):
        """Start a new recording session"""
        if self.is_recording:
            return
            
        self.recording_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_count = 0
        
        # Create recording directory
        recording_dir = os.path.join(self.output_dir, self.recording_id)
        os.makedirs(recording_dir, exist_ok=True)
        self.current_savepath = recording_dir
        # Initialize HDF5 file
        h5_path = os.path.join(recording_dir, "frames.h5")
        self.h5_file = h5py.File(h5_path, 'w')
        
        # Create HDF5 datasets with chunking and compression
        self.h5_file.create_dataset(
            "color_frames",
            shape=(0, self.height, self.width, 3),
            maxshape=(None, self.height, self.width, 3),
            dtype='uint8',
            chunks=(1, self.height, self.width, 3),
            compression="gzip",
            compression_opts=4
        )
        
        self.h5_file.create_dataset(
            "depth_frames",
            shape=(0, self.height, self.width),
            maxshape=(None, self.height, self.width),
            dtype='uint16',
            chunks=(1, self.height, self.width),
            compression="gzip",
            compression_opts=4
        )
        
        self.h5_file.create_dataset(
            "timestamps",
            shape=(0, 3),  # frame_timestamp, color_timestamp, depth_timestamp
            maxshape=(None, 3),
            dtype='float64'
        )
        
        # Store camera intrinsics as attributes
        intrinsics = self._create_intrinsics_dict()
        self.h5_file.attrs['camera_intrinsics'] = json.dumps(intrinsics)
        self.h5_file.attrs['fps'] = self.fps
        self.h5_file.attrs['width'] = self.width
        self.h5_file.attrs['height'] = self.height
        
        # Initialize video writers
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.video_writer = cv2.VideoWriter(
            os.path.join(recording_dir, "color.mp4"),
            fourcc, self.fps, (self.width, self.height)
        )
        self.depth_video_writer = cv2.VideoWriter(
            os.path.join(recording_dir, "depth_visualization.mp4"),
            fourcc, self.fps, (self.width, self.height)
        )
        
        # Save metadata
        metadata = {
            "recording_id": self.recording_id,
            "start_time": datetime.now().isoformat(),
            "fps": self.fps,
            "resolution": {"width": self.width, "height": self.height},
            "camera_intrinsics": intrinsics
        }
        with open(os.path.join(recording_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.is_recording = True
        print(f"Recording started. Directory: {recording_dir}")

    def stop_recording(self):
        """Stop the current recording session"""
        recording_dir = os.path.join(self.output_dir, self.recording_id)

        if not self.is_recording:
            return
            
        if self.h5_file is not None:
            self.h5_file.attrs['total_frames'] = self.frame_count
            self.h5_file.close()
            self.h5_file = None
            
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
        if self.depth_video_writer is not None:
            self.depth_video_writer.release()
            self.depth_video_writer = None
            
        # Update metadata
        if self.recording_id:
            metadata_path = os.path.join(self.output_dir, self.recording_id, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata.update({
                    "end_time": datetime.now().isoformat(),
                    "total_frames": self.frame_count
                })
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            
        self.is_recording = False
        print(f"Recording stopped. Frames captured: {self.frame_count}")

        # <--- Here is the important addition:
        if self.recording_stopped_callback is not None:
            self.recording_stopped_callback()

    def _normalize_depth_for_display(self, depth_image):
        """Convert depth image to colorized visualization"""
        normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = normalized_depth.astype(np.uint8)
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        return colored_depth

    def _append_frames(self, color_frame, depth_frame, frames):
        """Append color and depth frames to HDF5 and video files"""
        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Save to HDF5
        self.h5_file["color_frames"].resize((self.frame_count + 1, self.height, self.width, 3))
        self.h5_file["depth_frames"].resize((self.frame_count + 1, self.height, self.width))
        self.h5_file["timestamps"].resize((self.frame_count + 1, 3))
        
        self.h5_file["color_frames"][self.frame_count] = color_image
        self.h5_file["depth_frames"][self.frame_count] = depth_image
        self.h5_file["timestamps"][self.frame_count] = [
            frames.get_timestamp(),
            color_frame.get_timestamp(),
            depth_frame.get_timestamp()
        ]
        
        # Save to video files
        self.video_writer.write(color_image)
        
        # Create and save depth visualization
        depth_colormap = self._normalize_depth_for_display(depth_image)
        self.depth_video_writer.write(depth_colormap)
        
        # Flush HDF5 periodically
        if self.frame_count % 30 == 0:
            self.h5_file.flush()

    def capture_frames(self):
        """Main loop for capturing frames"""
        try:
            while True:
                # Wait for frameset
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy array for display
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Create displays
                depth_colormap = self._normalize_depth_for_display(depth_image)
                display_image = np.hstack((color_image, depth_colormap))
                
                # Show frames
                cv2.imshow('Color and Depth Frames', display_image)
                
                # If recording, save frames
                if self.is_recording:
                    self._append_frames(color_frame, depth_frame, frames)
                    self.frame_count += 1
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    if not self.is_recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
                elif key == ord('q'):
                    break
                
        finally:
            self.stop_recording()
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def get_current_savepath(self,):
        return self.current_savepath

    def get_current_recording_id(self,):
        return self.recording_id

    def get_ellm_studio_analysis(self, analyzer: VideoAnalyzer, rlef_uploader:VideoUploader, payload_filepath="payload.json"):
        payload = None
        try:
            #TODO: this is not the right way to do it, correct the videoanalyzer and the VideoUploader stuff in main.
            color_video_filepath = f'{self.current_savepath}/color.mp4'
            gcp_url = analyzer.upload_video_to_bucket("test1.mp4", color_video_filepath)
            video_annotations = analyzer.get_ellm_response()
            status_code = rlef_uploader.upload_to_rlef()
            print(f'status_code for RLEF Upload: {status_code}')
        except Exception as e:
            pass

    def send_annotations_to_rlef(self,):
        pass

    def prepare_annotations(self,):
        pass

if __name__ == "__main__":
    recorder = RealSenseRecorder()
    recorder.capture_frames() 