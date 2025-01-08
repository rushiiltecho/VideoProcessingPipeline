import streamlit as st
import cv2
import numpy as np
import h5py
import json
import os
from datetime import datetime
import pyrealsense2 as rs
from vdeo_analysis_ellm_sudio import VideoAnalyzer
from rlef_video_annotation import VideoUploader

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

        # Recording state
        self.recording_id = None
        self.frame_count = 0
        self.is_recording = False
        self.h5_file = None
        self.video_writer = None
        self.current_savepath = None
        self.depth_video_writer = None  # For visualization of depth data
        self.recording_stopped_callback = None

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

        # Initialize video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            os.path.join(recording_dir, "color.mp4"),
            fourcc, self.fps, (self.width, self.height)
        )
        self.depth_video_writer = cv2.VideoWriter(
            os.path.join(recording_dir, "depth_visualization.mp4"),
            fourcc, self.fps, (self.width, self.height)
        )

        # Metadata
        metadata = {
            "recording_id": self.recording_id,
            "start_time": datetime.now().isoformat(),
            "fps": self.fps,
            "resolution": {"width": self.width, "height": self.height}
        }
        with open(os.path.join(recording_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        self.is_recording = True
        st.write(f"Recording started. Directory: {recording_dir}")

    def stop_recording(self):
        """Stop the current recording session"""
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
        st.write(f"Recording stopped. Frames captured: {self.frame_count}")

    def _normalize_depth_for_display(self, depth_image):
        """Convert depth image to colorized visualization"""
        normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = normalized_depth.astype(np.uint8)
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        return colored_depth

    def _append_frames(self, color_frame, depth_frame, frames):
        """Append color and depth frames to HDF5 and video files"""
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
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        # Convert to numpy array for display
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Create displays
        depth_colormap = self._normalize_depth_for_display(depth_image)
        display_image = np.hstack((color_image, depth_colormap))

        return display_image, color_frame, depth_frame, frames

    def get_current_savepath(self):
        return self.current_savepath

    def get_current_recording_id(self):
        return self.recording_id

# Streamlit UI
def run_app():
    st.title("RealSense Recorder")

    recorder = RealSenseRecorder()
    
    # Control buttons
    if st.button("Start Recording"):
        recorder.start_recording()
    
    if st.button("Stop Recording"):
        recorder.stop_recording()

    # Display live stream
    frame_display, color_frame, depth_frame, frames = recorder.capture_frames()
    
    # Display images in Streamlit
    if frame_display is not None:
        st.image(frame_display, channels="BGR", use_column_width=True)

    # Upload functionality
    if st.button("Upload to Cloud"):
        analyzer = VideoAnalyzer()
        uploader = VideoUploader()
        recorder.get_ellm_studio_analysis(analyzer, uploader)

if __name__ == "__main__":
    run_app()
