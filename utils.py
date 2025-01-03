import json
import os
import re
import shutil
import tempfile
import h5py
import json
import numpy as np
import pyrealsense2 as rs
import vertexai
from google.cloud import storage


# vertexai.init(project=, location=, credentials=)

calib_matrix_x = np.array([
      [ 0.068, -0.986,  0.152, -0.108],
      [ 0.998,  0.065, -0.023,  0.0 ],
      [ 0.013,  0.153,  0.988, -0.044],
      [ 0.0,    0.0,    0.0,    1.0  ]
    ])

calib_matrix_y = np.array([
      [-0.47,   0.587,  -0.659,  0.73929],
      [ 0.877,  0.392,  -0.276, -0.16997],
      [ 0.096, -0.708,  -0.7,    0.86356],
      [ 0.0,    0.0,     0.0,    1.0    ]
    ])

# model = genai.GenerativeModel(
#   model_name='gemini-1.5-flash-002',
# )

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


def _transform_coordinates(point_xyz, calib_matrix_x=calib_matrix_x, calib_matrix_y=calib_matrix_y):
    """
    Transform point through both calibration matrices
    
    Args:
        point (list): [x, y, z] coordinates
        calib_x (list): First calibration matrix (4x4)
        calib_y (list): Second calibration matrix (4x4)
    
    Returns:
        list: Final transformed coordinates as regular floats [x, y, z]
    """
    # Convert inputs to numpy arrays
    point_array = np.array([*point_xyz, 1.0])
    calib_x_array = np.array(calib_matrix_x)
    calib_y_array = np.array(calib_matrix_y)
    
    # First transformation (X calibration)
    transformed_x = calib_x_array @ point_array
    if transformed_x[3] != 1.0:
        transformed_x = transformed_x / transformed_x[3]
    
    # Second transformation (Y calibration)
    transformed_y = calib_y_array @ transformed_x
    if transformed_y[3] != 1.0:
        transformed_y = transformed_y / transformed_y[3]
    
    # Convert to regular floats and return as list
    return [float(transformed_y[0]), float(transformed_y[1]), float(transformed_y[2])]


# =================== PARSING UTILS ===================
def parse_to_json(response):
    pattern = r"```json\s*(\{.*\})"
    match = re.search(pattern, response, re.DOTALL)
    json_content = ''
    if match:
        json_content = match.group(1)  # Extract the JSON string
    else:
        print("No valid JSON found.")
    
    return json_content
    

def transform_coordinates(point):
    """Transform coordinates using X and Y matrices."""
    B = np.eye(4)
    B[:3, 3] = point
    A = calib_matrix_y @ B @ np.linalg.inv(calib_matrix_x)
    transformed_point = A[:3, 3] * 1000
    return transformed_point[::-1]/1000

def parse_list_boxes(text:str):
  result = []
  for line in text.strip().splitlines():
    # Extract the numbers from the line, remove brackets and split by comma
    try:
      numbers = line.split('[')[1].split(']')[0].split(',')
    except:
      numbers =  line.split('- ')[1].split(',')

    # Convert the numbers to integers and append to the result
    result.append([int(num.strip()) for num in numbers])

  return result

def parse_list_boxes_with_label(text:str):
  text = text.split("```\n")[0]
  return json.loads(text.strip("```").strip("python").strip("json").replace("'", '"').replace('\n', '').replace(',}', '}'))

def upload_to_bucket(destination_blob_name, file_path):
    """
    Upload a video file to a Google Cloud Storage bucket.

    Args:
        destination_blob_name (str): Name of the blob in the bucket.
        video_file_path (str): Path to the video file.

    Input Format: str, str
    Output Format: str (gs:// URL)
    """
    credentials_file="ai-hand-service-acc.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
    storage_client = storage.Client()
    bucket = storage_client.bucket("video-analysing")
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    gs_url = f"gs://{bucket.name}/{destination_blob_name}"
    return gs_url


# PLOTTING UTILS ======================================================
# @title Plotting Utils
import json
import random
import io
from PIL import Image, ImageDraw
from PIL import ImageColor

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def plot_bounding_boxes(im, noun_phrases_and_positions):
    """
    Plots bounding boxes on an image with markers for each noun phrase, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        noun_phrases_and_positions: A list of tuples containing the noun phrases
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Iterate over the noun phrases and their positions
    for i, (noun_phrase, (y1, x1, y2, x2)) in enumerate(
        noun_phrases_and_positions):
        # Select a color from the list
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        abs_x1 = int(x1/1000 * width)
        abs_y1 = int(y1/1000 * height)
        abs_x2 = int(x2/1000 * width)
        abs_y2 = int(y2/1000 * height)

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
        )

        # Draw the text
        draw.text((abs_x1 + 8, abs_y1 + 6), noun_phrase, fill=color)

    # Display the image
    img.show()


def normalize_box(box, width=640, height=480):
    """
    Normalize bounding boxes from pixel coordinates to [0, 1] range.

    Args:
        boxes (list): List of bounding boxes in [ymin, xmin, ymax, xmax] format.
        width (int): Image width.
        height (int): Image height.

    Returns:
        list: Normalized bounding boxes in [ymin, xmin, ymax, xmax] format.
    """

    ymin, xmin, ymax, xmax = box
    normalized_box = [ xmin / 1000*width, ymin / 1000*height, xmax / 1000*width, ymax / 1000*height]
    return normalized_box

# REGION SELECTOR UTILS ============================================================


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