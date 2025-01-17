import base64
import json
import os
from typing import Dict, Optional, AnyStr
import json_repair
import requests

from vdeo_analysis_ellm_sudio import VideoAnalyzer


class VideoUploader:
    def __init__(self, filepath:Optional[str]= None, video_annotations:Optional[dict]=None):
        self.video_filepath = filepath
        self.video_annotations = video_annotations

    def convert_video(self, input_path, output_path):
        os.system(f"ffmpeg -i '{input_path}' -c:v libx264 '{output_path}'")

    def bytes(self, file_path):
        with open(file_path, 'rb') as binary_file:
            binary_content = binary_file.read()
            encoded_content = base64.b64encode(binary_content)
            return encoded_content

    def upload_to_rlef(self, rlef_url ,video_filepath, video_annotations, csv_filepath):
        self.video_annotations = video_annotations
        self.video_filepath = video_filepath
        converted_filepath = f'{self.video_filepath}_converted.mp4'
        self.convert_video(self.video_filepath, converted_filepath)

        payload = {
            'model': '67695dc462913593227a4227',
            'status': 'backlog',
            'csv': 'self.bytes(csv_filepath)', #TODO: get the csv file as text here
            'label': 'object_grab',
            'tag': 'loaner boxes',
            'prediction': 'predicted',
            'confidence_score': '100',
            'videoAnnotations': self.generate_video_annotations(video_annotations)
        }

        files = {
            'resource': (converted_filepath, open(converted_filepath, 'rb'))
        }

        response = requests.post(
            rlef_url , 
            headers={},
            data=payload,
            files=files
        )

        print(f"RLEF RESPONSE STATUS: =========== {response.status_code}")
        processed_response = json.loads(json_repair.repair_json(response.text))
        # print(f"RLEF RESPONSE TEXT: =========== {processed_response}")

        return response.status_code, processed_response


    def upload_to_rlef_train(self, rlef_url ,video_filepath, video_annotations, model='678a262dc441e0b2c81a9686'):
        self.video_annotations = video_annotations
        self.video_filepath = video_filepath
        converted_filepath = f'{self.video_filepath.split(".")[0]}.mp4'
        self.convert_video(self.video_filepath, converted_filepath)

        payload = {
            'model': model,
            'status': 'backlog',
            'csv': 'self.bytes(csv_filepath)', #TODO: get the csv file as text here
            'label': 'object_grab',
            'tag': 'loaner boxes',
            'prediction': 'predicted',
            'confidence_score': '100',
            'videoAnnotations': self.generate_video_annotations(video_annotations)
        }

        files = {
            'resource': (converted_filepath, open(converted_filepath, 'rb'))
        }

        response = requests.post(
            rlef_url , 
            headers={},
            data=payload,
            files=files
        )

        print(f"RLEF RESPONSE STATUS: =========== {response.status_code}")
        processed_response = json.loads(json_repair.repair_json(response.text))
        # print(f"RLEF RESPONSE TEXT: =========== {processed_response}")

        return response.status_code, processed_response


    def generate_video_annotations(self, video_annotations):
        video_annotations_list = []
        print(video_annotations)
        try:
            for i in video_annotations.keys():
                if isinstance(video_annotations[i], list) and all(isinstance(item, dict) for item in video_annotations[i]):
                    for j in range(len(video_annotations[i])):
                        # print(self.video_annotations[i][j], i)
                        video_annotations_list.append({
                            "label": i,
                            "tag": video_annotations[i][j]['object_name'],
                            "annotationPrediction": {
                                "startTimeInSeconds": self.convert_time_to_seconds(video_annotations[i][j]['start_time']),
                                "endTimeInSeconds": self.convert_time_to_seconds(video_annotations[i][j]['end_time'])
                            }
                        })
        except Exception as e:
            print("Annotations inapporpriate")

        finally:
            # print("VIDEO ANNOTATIONS",json_repair.repair_json(str(video_annotations_list)))
            return str(video_annotations_list).replace("'", '"')


    def _deprecated_generate_video_annotations(self):
        video_annotations_list = []
        for i in self.video_annotations.keys():
            if isinstance(self.video_annotations[i], list) and all(isinstance(item, dict) for item in self.video_annotations[i]):
                for j in range(len(self.video_annotations[i])):
                    # print(self.video_annotations[i][j], i)
                    video_annotations_list.append({
                        "label": i,
                        "tag": self.video_annotations[i][j]['object_name'],
                        "annotationPrediction": {
                            "startTimeInSeconds": self.convert_time_to_seconds(self.video_annotations[i][j]['start_time']),
                            "endTimeInSeconds": self.convert_time_to_seconds(self.video_annotations[i][j]['end_time'])
                        }
                    })

        print("VIDEO ANNOTATIONS",json_repair.repair_json(str(video_annotations_list)))
        return str(video_annotations_list).replace("'", '"')

    # Function to get the signed URL from the RLEF API
    def get_signed_url(self, rlef_url= "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/uploadHdf5File", resource_id =None, hdf5_filename=None):
        url = rlef_url if rlef_url else "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/uploadHdf5File"
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
    def upload_csv_file(self, signed_url, hdf5_filepath):
        headers = {"Content-Type": "application/octet-stream"}
        
        with open(hdf5_filepath, 'rb') as file_data:
            response = requests.put(signed_url, headers=headers, data=file_data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response from the server: {response.text}")

    # Main function to coordinate the process
    def process_and_upload_csv(self, video_bucket_id, csv_filepath,csv_filename):
        #THINK
        # Step 3: Get signed URL for uploading the HDF5 file
        signed_url = self.get_signed_url(video_bucket_id, csv_filename)
        print(f"Signed URL: {signed_url}")
        if signed_url:
            print(f"Signed URL: {signed_url}")
            
            # Step 4: Upload the HDF5 file using the signed URL
            self.upload_csv_file(signed_url, csv_filepath)
        else:
            print("Failed to obtain the signed URL.")

    def convert_time_to_seconds(self, time):
        time_parts = time.split(':')
        if len(time_parts) == 3:
            h, m, s = map(int, time_parts)
            return h * 3600 + m * 60 + s
        elif len(time_parts) == 2:
            m, s = map(int, time_parts)
            return m * 60 + s
        else:
            raise ValueError("Invalid time format")

sample_video_annotations = {
    "overall_task_name": "Object Rearrangement",
    "objects": [
        "Mug",
        "Water Bottle",
        "Soda Can"
    ],
    "picking_up": [
        {
            "start_time": "00:01",
            "end_time": "00:02",
            "object_name": "Mug",
            "notes": ""
        },
        {
            "start_time": "00:03",
            "end_time": "00:04",
            "object_name": "Water Bottle",
            "notes": ""
        },
        {
            "start_time": "00:06",
            "end_time": "00:07",
            "object_name": "Soda Can",
            "notes": ""
        }
    ],
    "placing": [
        {
            "start_time": "00:02",
            "end_time": "00:03",
            "object_name": "Mug",
            "notes": "Placed next to soda can"
        },
        {
            "start_time": "00:04",
            "end_time": "00:05",
            "object_name": "Water Bottle",
            "notes": "Placed in original mug's spot"
        },
        {
            "start_time": "00:07",
            "end_time": "00:08",
            "object_name": "Soda Can",
            "notes": "Placed in original water bottle spot"
        }
    ]
}

if __name__ == "__main__":
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/"
    headers = {
        "Content-Type": "application/json"
    }
    #TODO: make the payload customizable.
    # enable change in video links, agents, etc.
    payload = None
    # Load the payload from a JSON file
    with open("payload.json", "r") as file:
        payload = json.load(file)

    print(f"================ PAYLOAD ================ +\n{payload['question']}\n================ PAYLOAD ================")
    analyzer = VideoAnalyzer(payload=payload)
    video_file_path = 'C:\\Users\\Rushiil Bhatnagar\\Downloads\\object_detection\\object_detection\\videos\\pick_and_place_1.mp4'
    gcp_url = analyzer.upload_video_to_bucket("test1.mp4", video_file_path)
    # video_annotations = analyzer.get_ellm_response()
    video_annotations = analyzer.get_gemini_response(gcp_url=gcp_url)
    uploader = VideoUploader(video_file_path, video_annotations)
    status_code = uploader.upload_to_rlef(url, video_file_path, video_annotations)
    print(status_code)