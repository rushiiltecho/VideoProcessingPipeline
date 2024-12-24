import time
import pyrealsense2 as rs
import cv2
import numpy as np

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure the streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Define the codec and create VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
out = None
is_recording = False

try:
    while True:
        # Wait for a frame and get the frameset
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Display the color image
        cv2.imshow('Color Frame', color_image)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        # Start/stop recording on 'r' key press
        if key == ord('r'):
            if not is_recording:
                # Start recording
                filename = f'output_{int(time.time())}.mp4'
                out = cv2.VideoWriter(filename, fourcc, 30.0, (color_image.shape[1], color_image.shape[0]))
                
                # Debugging check: Confirm if the VideoWriter was successfully opened
                if not out.isOpened():
                    print("Error: VideoWriter failed to initialize.")
                    break
                
                print(f"Recording started... Saving to {filename}")
                is_recording = True
            else:
                # Stop recording
                out.release()
                print("Recording stopped.")
                is_recording = False

        # Write the color frame to the video file if recording
        if is_recording:
            out.write(color_image)

        # Exit on 'q' key press
        if key == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    # Release the VideoWriter if it's still open
    if out is not None:
        out.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
