import base64
import json
import os
from typing import Dict, List
import json_repair
import requests

from vdeo_analysis_ellm_sudio import VideoAnalyzer


class VideoUploader:
    def __init__(self, filepath, video_annotations):
        self.filepath = filepath
        self.video_annotations = video_annotations

    def convert_video(self, input_path, output_path):
        os.system(f"ffmpeg -i '{input_path}' -c:v libx264 '{output_path}'")

    def bytes(self, file_path):
        with open(file_path, 'rb') as binary_file:
            binary_content = binary_file.read()
            encoded_content = base64.b64encode(binary_content)
            return encoded_content

    def upload_to_rlef(self):
        converted_filepath = self.filepath

        self.convert_video(self.filepath, converted_filepath)

        url = 'https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/'

        payload = {
            'model': '67695dc462913593227a4227',
            'status': 'backlog',
            'csv': 'csv',
            'label': 'object_grab',
            'tag': 'loaner boxes',
            'prediction': 'predicted',
            'confidence_score': '100',
            'videoAnnotations': self.generate_video_annotations()
        }

        files = {
            'resource': (converted_filepath, open(converted_filepath, 'rb'))
        }

        response = requests.post(url, headers={}, data=payload, files=files)

        print(response.text)
        return response.status_code

    def generate_video_annotations(self):
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

    # def generate_video_annotations(self):
    #     video_annotations = []
    #     for picking_up in self.video_annotations['picking_up']:
    #         video_annotations.append({
    #             "label": "picking_up",
    #             "tag": picking_up['object_name'],
    #             "annotationPrediction": {
    #                 "startTimeInSeconds": self.convert_time_to_seconds(picking_up['start_time']),
    #                 "endTimeInSeconds": self.convert_time_to_seconds(picking_up['end_time'])
    #             }
    #         })
    #     for placing in self.video_annotations['placing']:
    #         video_annotations.append({
    #             "label": "placing",
    #             "tag": placing['object_name'],
    #             "annotationByExpert": {
    #                 "startTimeInSeconds": self.convert_time_to_seconds(placing['start_time']),
    #                 "endTimeInSeconds": self.convert_time_to_seconds(placing['end_time']),
    #                 "approvalStatus": "approved"
    #             }
    #         })
    #     return str(video_annotations).replace("'", '"')

    def convert_time_to_seconds(self, time):
        m, s = map(int, time.split(':'))
        return m * 60 + s

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
    url = "https://dev-egpt.techo.camp/predict"
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
    video_file_path = 'C:\\Users\\Rushiil Bhatnagar\\Downloads\\object_detection\\object_detection\\videos\\pouring_1.mp4'
    gcp_url = analyzer.upload_video_to_bucket("test1.mp4", video_file_path)
    video_annotations = analyzer.get_ellm_response()
    uploader = VideoUploader(video_file_path, video_annotations)
    status_code = uploader.upload_to_rlef()
    print(status_code)