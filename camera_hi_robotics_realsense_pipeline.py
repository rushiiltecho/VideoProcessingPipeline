from datetime import datetime
import json
import os
import time
import zipfile
import cv2
import h5py
import numpy as np
from hi_robotics.vision_ai.cameras.intel_realsense_camera import IntelRealSenseCamera



class RealSenseRecorder:
    def __init__(self, camera: IntelRealSenseCamera, output_dir:str= "recordings" ,fps= 10):
        self.camera = camera
        self.rgb_video = None
        self.depth_video = None

        # Camera Intrinsics Parameters:
        self.intrinsics = camera.get_intrinsics(depth=True)
        self.depth_intrinsics = self.intrinsics['depth_intrinsics']
        self.color_intrinsics = self.intrinsics['color_intrinsics']
        self.width = self.depth_intrinsics.width
        self.height = self.depth_intrinsics.height
        self.fps = fps if fps else 10
        self.depth_scale =  0.001
        
        # Recording Status Management
        self.recording_id = None
        self.h5_file = None
        self.color_video_writer = None
        self.depth_video_writer = None
        
        self.current_savepath = None
        self.frame_count = 0
        self.is_recording = False
        self.recording_stopped_callback = None

        # Recording Save Management
        self.output_dir = output_dir

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
            "depth_scale": self.depth_scale if self.depth_scale else 0.001
        }
    
    def start_recording(self, recording_id:str = None):
        """
        Start recording the camera feed to a video file.
        """
        if self.is_recording:
            return
            
        self.recording_id = recording_id if recording_id else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_count = 0
        
        # Create recording directory
        recording_dir = os.path.join(self.output_dir, self.recording_id)
        os.makedirs(recording_dir, exist_ok=True)
        self.current_savepath = recording_dir
        self.data_collection_directory_creation()
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
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            os.path.join(self.current_savepath, "color.mp4"),
            fourcc, self.fps, (self.width, self.height)
        )
        self.depth_video_writer = cv2.VideoWriter(
            os.path.join(self.current_savepath, "depth_visualization.mp4"),
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
        with open(os.path.join(self.current_savepath, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.is_recording = True
        print(f"Recording started. Directory: {self.current_savepath}")

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
        self.save_zips()
        print(f"Recording stopped. Frames captured: {self.frame_count}")

        # [IGNORE]<--- Here is the addition:
        if self.recording_stopped_callback is not None:
            self.recording_stopped_callback()

    def _normalize_depth_for_display(self, depth_image):
        """Convert depth image to colorized visualization"""
        normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        normalized_depth = normalized_depth.astype(np.uint8)
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        return colored_depth

    def _append_frames(self, color_frame, depth_frame):
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
            color_frame.get_timestamp(),
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

    def capture_frames(self,):
        """Capture frames from the camera and save them to the recording"""
        try:
            while True:
                rgb_frame, depth_frame = self.camera.get_frames()
                color_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                depth_colormap = self._normalize_depth_for_display(depth_image)
                display_image = np.hstack((color_image, depth_colormap))
                save_frame_count_diff = self.frame_count

                cv2.imshow("RGB-D Frame", display_image)

                if self.is_recording:
                    prev_time = time.time()

                    self._append_frames(rgb_frame, depth_frame)
                    self.frame_count += 1
                    
                    current_time = time.time()
                    if current_time - prev_time < 1.0 / self.fps:
                        # Save RGB image:
                        rgb_path = f"{self.current_savepath}/rgb_images_data_collection/image_{self.frame_count}.jpg"
                        cv2.imwrite(rgb_path, color_image)

                        # Save depth image as .npy:
                        depth_path = f"{self.current_savepath}/depth_images_data_collection/image_{self.frame_count}.npy"
                        np.save(depth_path, depth_image)

                        # Increment image counter and update last capture time
                        print(f"Saved : {self.frame_count}")
                        prev_time = current_time

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
            self.camera.release_camera()
            cv2.destroyAllWindows()

    def save_zips(self,):
        # Create timestamp for unique zip names
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create zip archives
        rgb_zip_name = f"{self.current_savepath}/archives/rgb_images_data_collection.zip"
        depth_zip_name = f"{self.current_savepath}/archives/depth_images_data_collection.zip"

        
        print("Creating zip archives...")
        self.create_zip_archive(f'{self.current_savepath}/archives', rgb_zip_name)
        self.create_zip_archive(f'{self.current_savepath}/archives', depth_zip_name)
        
        print("Archives created successfully!")

    def get_current_savepath(self,):
        return self.current_savepath
    
    def get_current_recording(self,):
        return self.recording_id
    
    def create_zip_archive(self, source_dir, zip_name):
        """Create a zip file from a directory"""
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)
        print(f"Created zip archive: {zip_name}")
    
    def data_collection_directory_creation(self,):
        """Collect data from the camera and save it to a zip archive"""
        # Create directories if they don't exist
        os.makedirs(f'{self.current_savepath}/rgb_images_data_collection', exist_ok=True)
        os.makedirs(f'{self.current_savepath}/depth_images_data_collection', exist_ok=True)
        os.makedirs(f'{self.current_savepath}/videos_data_collection', exist_ok=True)
        os.makedirs(f'{self.current_savepath}/archives', exist_ok=True)

        

def sample_function():
    camera= IntelRealSenseCamera()

    try:
        print(camera.get_intrinsics(depth=True)['depth_intrinsics'])
        print(camera.get_intrinsics(depth=True)['depth_intrinsics'].width)
        while True:
            # Capture frames from the camera
            rgb_frame, depth_frame = camera.get_frames()
            color_image = np.asanyarray(rgb_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_image_colored = cv2.cvtColor(depth_image_normalized, cv2.COLOR_GRAY2BGR)
            display_image = np.hstack((color_image, depth_image_colored))

            cv2.imshow("RGB Image", display_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    
    finally:
        camera.release_camera()
        cv2.destroyAllWindows()
        print('Camera released and windows destroyed')


if __name__ == "__main__": 
    camera = IntelRealSenseCamera()
    recorder = RealSenseRecorder(camera)
    recorder.capture_frames()
    print(recorder.get_current_savepath())
    print(recorder.get_current_recording())
    # camera.release_camera()
    # cv2.destroyAllWindows()
    # print('Camera released and windows destroyed')