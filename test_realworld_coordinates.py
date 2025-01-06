import pyrealsense2 as rs
import numpy as np
import cv2

from utils import transform_coordinates

def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable only color stream at 640x480
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get color sensor intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    return pipeline, color_intrinsics

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        intrinsics = param['intrinsics']
        
        # Get normalized coordinates
        nx = (x - intrinsics.ppx) / intrinsics.fx
        ny = (y - intrinsics.ppy) / intrinsics.fy
        
        # Print pixel coordinates and normalized coordinates
        print(f"Pixel coordinates: ({x}, {y})")
        # Deproject pixel to 3D point in camera coordinates
        depth = 1.0  # Assuming a depth of 1 meter for demonstration
        point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
        transformed_3d = transform_coordinates(point_3d)
        print(f"3D coordinates: ({transformed_3d[0]:.4f}, {transformed_3d[1]:.4f}, {transformed_3d[2]:.4f}) meters")
        # print(f"Normalized coordinates: ({nx:.4f}, {ny:.4f})")
        
        # # If you want to project to 3D space at a specific Z distance (in meters)
        # z = 1.0  # Example: 1 meter distance
        # x3d = nx * z
        # y3d = ny * z
        
        # print(f"3D coordinates at {z}m distance: ({x3d:.4f}, {y3d:.4f}, {z:.4f}) meters")
        
        # Print distortion coefficients
        if intrinsics.model == rs.distortion.inverse_brown_conrady:
            print("Distortion coefficients:", intrinsics.coeffs)

def main():
    # Initialize RealSense and get intrinsics
    pipeline, color_intrinsics = initialize_realsense()
    
    # Create window and set mouse callback
    window_name = 'RealSense Color Stream'
    cv2.namedWindow(window_name)
    
    # Print camera intrinsics information
    print("\nCamera Intrinsics:")
    print(f"Focal Length (fx, fy): ({color_intrinsics.fx}, {color_intrinsics.fy})")
    print(f"Principal Point (ppx, ppy): ({color_intrinsics.ppx}, {color_intrinsics.ppy})")
    print(f"Distortion Model: {color_intrinsics.model}")
    
    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Set mouse callback with intrinsics as parameter
            param = {'intrinsics': color_intrinsics}
            cv2.setMouseCallback(window_name, mouse_callback, param)

            # Show color image
            cv2.imshow(window_name, color_image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()