import json
import os
import re
import shutil
import tempfile
import h5py
import json
import numpy as np
import pyrealsense2 as rs

def convert_video(input_path, output_path):
    # Create a temporary file
    temp_output_path = tempfile.mktemp(suffix='.mp4')
    
    # Run the ffmpeg command to convert the video
    os.system(f"ffmpeg -i '{input_path}' -c:v libx264 '{temp_output_path}'")
    
    # Check if the conversion was successful
    if os.path.exists(temp_output_path):
        # Remove the original file
        os.remove(output_path)
        # Rename the temporary file to the original output path
        shutil.move(temp_output_path, output_path)
    else:
        print("Conversion failed.")

def parse_to_json(response):
    pattern = r"```json\s*(\{.*\})"
    match = re.search(pattern, response, re.DOTALL)
    json_content = ''
    if match:
        json_content = match.group(1)  # Extract the JSON string
    else:
        print("No valid JSON found.")
    
    return json_content

def get_pixel_3d_coordinates(recording_dir, time_seconds, pixel_x, pixel_y):
    """
    Get the 3D coordinates (X, Y, Z) of a specific pixel at a specific time in the video
    
    Args:
        recording_dir (str): Path to the recording directory
        time_seconds (float): Time in seconds from the start of the video
        pixel_x (int): X coordinate of the pixel
        pixel_y (int): Y coordinate of the pixel
    
    Returns:
        tuple: ((X, Y, Z) coordinates in meters, actual_time in seconds)
    """
    # Open the H5F file
    h5_path = f"{recording_dir}/frames.h5"
    with h5py.File(h5_path, 'r') as h5_file:
        # Get timestamps array
        timestamps = h5_file['timestamps'][:]
        
        # Convert timestamps to seconds from start
        frame_times = timestamps[:, 0] - timestamps[0, 0]
        
        # Find the closest frame to the requested time
        closest_frame = np.argmin(np.abs(frame_times - time_seconds))
        
        # Get the depth frame
        depth_frame = h5_file['depth_frames'][closest_frame]
        
        # Get actual timestamp of the frame we're using
        actual_time = frame_times[closest_frame]
        print(f"Using frame at {actual_time:.3f} seconds (requested: {time_seconds:.3f} seconds)")
        
        # Get camera intrinsics from metadata
        intrinsics_str = h5_file.attrs['camera_intrinsics']
        intrinsics_dict = json.loads(intrinsics_str)
        depth_scale = intrinsics_dict['depth_scale']
        
        # Create RealSense intrinsics object
        depth_intrinsics = rs.intrinsics()
        d_intr = intrinsics_dict['depth_intrinsics']
        depth_intrinsics.width = h5_file.attrs['width']
        depth_intrinsics.height = h5_file.attrs['height']
        depth_intrinsics.ppx = d_intr['ppx']
        depth_intrinsics.ppy = d_intr['ppy']
        depth_intrinsics.fx = d_intr['fx']
        depth_intrinsics.fy = d_intr['fy']
        depth_intrinsics.model = rs.distortion.brown_conrady
        depth_intrinsics.coeffs = d_intr['coeffs']
        
        # Get depth value for the pixel (in millimeters)
        depth_value = depth_frame[pixel_y, pixel_x]
        
        # Convert depth to meters
        depth_in_meters = depth_value * depth_scale
        
        # Deproject pixel to 3D point
        point_3d = rs.rs2_deproject_pixel_to_point(
            depth_intrinsics,
            [pixel_x, pixel_y],
            depth_in_meters
        )
        
        return point_3d, actual_time

if __name__ == "__main__":
    # Example parameters
    recording_dir = "recordings/20241227_205319"
    time_second = 5  # 5 seconds into the video
    pixel_x = 320  # Center X (assuming 640x480 resolution)
    pixel_y = 240  # Center Y (assuming 640x480 resolution)
    
    try:
        coords, actual_time = get_pixel_3d_coordinates(recording_dir, time_second, pixel_x, pixel_y)
        print(f"3D coordinates at {actual_time:.3f} seconds (in meters):")
        print(f"X: {coords[0]:.3f}")
        print(f"Y: {coords[1]:.3f}")
        print(f"Z: {coords[2]:.3f}")
    except Exception as e:
        print(f"Error getting coordinates: {str(e)}")
# if __name__== "__main__":
#     string= '''
#     ```json
#         {
#         "objects": [
#             "Coca-Cola can",
#             "Black mug",
#             "Blue water bottle"
#         ],
#         "Picking up": [
#             {
#             "start_time": "00:00",
#             "end_time": "00:03",
#             "object_name": "Coca-Cola can",
#             "notes": "Human hand picks up the Coca-Cola can."
#             },
#             {
#             "start_time": "00:03",
#             "end_time": "00:06",
#             "object_name": "Black mug",
#             "notes": "Human hand picks up the black mug after putting down the can."
#             },
#             {
#             "start_time": "00:08",
#             "end_time": "00:11",
#             "object_name": "Blue water bottle",
#             "notes": "Human hand picks up the blue water bottle."
#             }
#         ],
#         "Placing down": [
#             {
#             "start_time": "00:03",
#             "end_time": "00:04",
#             "object_name": "Coca-Cola can",
#             "notes": "Human hand places the Coca-Cola can back on the table."
#             },
#             {
#             "start_time": "00:06",
#             "end_time": "00:08",
#             "object_name": "Black mug",
#             "notes": "Human hand places the black mug back on the table."
#             }
#         ]
#         }
#         ```
#     '''

#     parse_to_json(string)