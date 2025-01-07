from hi_robotics.hi_robotics.vision_ai.cameras.intel_realsense_camera import IntelRealSenseCamera
import cv2
import csv
import os
import numpy as np
import time
import zipfile
from datetime import datetime

camera = IntelRealSenseCamera()

def create_zip_archive(source_dir, zip_name):
    """Create a zip file from a directory"""
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"Created zip archive: {zip_name}")

def collect_data():
    # Create directories if they don't exist
    os.makedirs("rgb_images", exist_ok=True)
    os.makedirs("depth_images", exist_ok=True)
    os.makedirs("videos", exist_ok=True)
    os.makedirs("archives", exist_ok=True)

    # Initialize counters
    image_counter = 0

    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rgb_video = cv2.VideoWriter('videos/rgb_recording.mp4', 
                               fourcc, 10.0, 
                               (640, 480))
    depth_video = cv2.VideoWriter('videos/depth_recording.mp4', 
                                 fourcc, 10.0, 
                                 (640, 480))

    # Calculate time interval for 10 FPS
    interval = 1/10  # seconds between frames
    last_capture_time = time.time()

    try:
        while True:
            # Capture frames from the camera
            rgb_frame, depth_frame = camera.get_frames()
            color_image = np.asanyarray(rgb_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Normalize depth image for better visualization
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Convert depth image to 3-channel for video writing
            depth_image_colored = cv2.cvtColor(depth_image_normalized, cv2.COLOR_GRAY2BGR)

            # Write frames to video
            rgb_video.write(color_image)
            depth_video.write(depth_image_colored)

            # Display both the RGB and depth frames
            cv2.imshow("RGB Frame", color_image)
            cv2.imshow("Depth Frame", depth_image_normalized)

            current_time = time.time()
            # Check if it's time to capture a new frame (10 FPS)
            if current_time - last_capture_time >= interval:
                # Save RGB image
                rgb_path = f"rgb_images/image_{image_counter}.jpg"
                cv2.imwrite(rgb_path, color_image)

                # Save depth image as .npy
                depth_path = f"depth_images/image_{image_counter}.npy"
                np.save(depth_path, depth_image)

                # Increment image counter and update last capture time
                print(f"Saved : {image_counter}")
                image_counter += 1
                last_capture_time = current_time

            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        # Clean up
        rgb_video.release()
        depth_video.release()
        cv2.destroyAllWindows()

        # Create timestamp for unique zip names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create zip archives
        rgb_zip_name = f"archives/rgb_images_{timestamp}.zip"
        depth_zip_name = f"archives/depth_images_{timestamp}.zip"
        
        print("Creating zip archives...")
        create_zip_archive("rgb_images", rgb_zip_name)
        create_zip_archive("depth_images", depth_zip_name)
        
        print("Archives created successfully!")
        
        # Optionally, clean up the original image directories
        # Uncomment these lines if you want to delete the original images after zipping
        # import shutil
        # shutil.rmtree("rgb_images")
        # shutil.rmtree("depth_images")
        # print("Cleaned up original image directories")

if __name__ == "__main__":
    collect_data()