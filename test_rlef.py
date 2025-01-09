import os
import requests
import json
  # Adjust based on your actual file structure

# Function to convert video using ffmpeg
def convert_video(input_path, output_path):
    os.system(f"ffmpeg -i '{input_path}' -c:v libx264 '{output_path}'")

# Function to upload video to the RLEF API
def upload_video(filepath):
    # Convert the video file
    converted_filepath = filepath.replace(".mp4", "_converted.mp4")
    convert_video(filepath, converted_filepath)

    # API endpoint
    url = 'https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/'
    
    # Construct the videoAnnotations JSON
    video_annotations = [
        {
            "label": "approaching_object",
            "tag": "31278965",
            "annotationPrediction": {
                "startTimeInSeconds":0,
                "endTimeInSeconds": 1
            }
        },
        {
            "label": "grabbing_object",
            "tag": "31278966",
            "annotationPrediction": {
                "startTimeInSeconds": 1,
                "endTimeInSeconds": 2
            }
        },
        {
            "label": "moving_object",
            "tag": "31278967",
            "annotationPrediction": {
                "startTimeInSeconds":2,
                "endTimeInSeconds": 3
            }
        }
    ]

    # Payload data for the request
    payload = {
        'model': '67695dc462913593227a4227',
        'status': 'backlog',
        'csv': "csv",
        'label': 'object_grab',
        'tag': 'loaner boxes',
        'prediction': 'predicted',
        'confidence_score': '100',
        'videoAnnotations': json.dumps(video_annotations)  # Convert list to JSON string
    }
    
    # Upload the converted file
    try:
        with open(converted_filepath, 'rb') as f:
            files = {'resource': (converted_filepath, f)}
            response = requests.post(url, headers={}, data=payload, files=files)
        
        # Print response info
        print(f"Status Code: {response.status_code}")
        response_json = response.json() if response.status_code == 200 else None
        return response_json
    
    except FileNotFoundError:
        print(f"File not found: {converted_filepath}")
        return None

# Function to get the signed URL from the RLEF API
def get_signed_url(resource_id, hdf5_filename):
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/uploadHdf5File"
    form_data = {
        "resourceId": resource_id,
        "hdf5FileName": hdf5_filename
    }
    response = requests.put(url, data=form_data)
    
    if response.status_code == 200:
        try:
            response_dict = response.json()
            return response_dict.get("hdf5FileSignedUrlForUpload")
        except json.JSONDecodeError:
            print("Error: Response is not valid JSON.")
            return None
    else:
        print(f"Failed to get signed URL. Status code: {response.status_code}")
        return None

# Function to upload the HDF5 file to the signed URL
def upload_hdf5_file(signed_url, hdf5_filepath):
    headers = {"Content-Type": "application/octet-stream"}
    
    with open(hdf5_filepath, 'rb') as file_data:
        response = requests.put(signed_url, headers=headers, data=file_data)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response from the server: {response.text}")

# Main function to coordinate the process
def process_and_upload(filepath, hdf5_filepath, hdf5_filename):
    # Step 1: Extract frame information from the HDF5 file
  
    
    # Step 2: Upload video and get response
    response_json = upload_video(filepath)
    
    if response_json and '_id' in response_json:
        video_bucket_id = response_json['_id']
        print(f"Extracted Video Bucket ID: {video_bucket_id}")
        
        # Step 3: Get signed URL for uploading the HDF5 file
        signed_url = get_signed_url(video_bucket_id, hdf5_filename)
        
        if signed_url:
            print(f"Signed URL: {signed_url}")
            
            # Step 4: Upload the HDF5 file using the signed URL
            upload_hdf5_file(signed_url, hdf5_filepath)
        else:
            print("Failed to obtain the signed URL.")
    else:
        print("Video upload failed or '_id' not found in the response.")

# Execute the entire process
if __name__ == "__main__":
    video_path = "/home/ai_hand/Downloads/main_flow_dec27_DEMO/recordings/Recorded_Demo/color.mp4"
    hdf5_path = "recordings/Recorded_Demo/hamer_output/predictions_hamer_sample.csv"
    hdf5_filename = "predictions_hamer_sample.csv"
    
    process_and_upload(video_path, hdf5_path, hdf5_filename)