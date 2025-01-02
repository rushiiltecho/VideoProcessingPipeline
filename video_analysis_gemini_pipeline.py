"""
Author: RUSHIIL BHATNAGAR
E-mail: rushiil.bhatnagar@techolution.com
Description:
This code is part of a video analysis system that utilizes Google Cloud's Vertex AI to generate text descriptions of video content.
The system consists of the following components:

1. VideoUploader: Uploads video files to a Google Cloud Storage bucket.
2. TextGenerator: Uses a Vertex AI generative model to generate text descriptions of the video content.
3. ResponseParser: Parses the response text from the generative model into a JSON object.
4. VideoSplitter: Splits the video into parts based on the actions and objects identified in the JSON object.

The system takes a video file as input, uploads it to a Google Cloud Storage bucket, generates a text description of the video content,
parses the response text into a JSON object, and splits the video into parts based on the actions and objects identified in the JSON object.
The JSON object is then saved to a file.

The code is written in Python and utilizes the Google Cloud Client Library and the OpenCV library for video processing.

Usage:
To use this code, simply run the main function and provide the necessary input parameters, such as the video file path and the Vertex AI model name.
The code will then upload the video file to a Google Cloud Storage bucket, generate a text description of the video content,
parse the response text into a JSON object, and split the video into parts based on the actions and objects identified in the JSON object.
The JSON object will then be saved to a file.

Note:
This code requires a Google Cloud account and a Vertex AI model to function.
It also requires the necessary dependencies, such as the Google Cloud Client Library and the OpenCV library, to be installed.
"""
import json
import logging
import re
import vertexai
from google.cloud import storage
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import os
from utils import parse_to_json
import cv2

class VideoAnalyzer:
    def __init__(self, credentials_file, project, location):
        """
        Initialize the VideoAnalyzer class.

        Args:
            credentials_file (str): Path to the Google application credentials file.
            project (str): Google Cloud project ID.
            location (str): Google Cloud location.

        Input Format: None
        Output Format: None
        """
        self.credentials_file = credentials_file
        self.project = project
        self.location = location
        self.credentials = service_account.Credentials.from_service_account_file(credentials_file)
        vertexai.init(project=project, location=location, credentials=self.credentials)

    def upload_video_to_bucket(self, destination_blob_name, video_file_path):
        """
        Upload a video file to a Google Cloud Storage bucket.

        Args:
            destination_blob_name (str): Name of the blob in the bucket.
            video_file_path (str): Path to the video file.

        Input Format: str, str
        Output Format: str (gs:// URL)
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_file
        storage_client = storage.Client()
        bucket = storage_client.bucket("video-analysing")
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(video_file_path)
        gs_url = f"gs://{bucket.name}/{destination_blob_name}"
        return gs_url

    def generate_text(self, vision_model_name, gcp_url, prompt):
        """
        Generate text using a Vertex AI generative model.

        Args:
            vision_model_name (str): Name of the Vertex AI generative model.
            gcp_url (str): gs:// URL of the video file.
            prompt (str): Prompt for the generative model.

        Input Format: str, str, str
        Output Format: str (generated text)
        """
        vision_model = GenerativeModel(vision_model_name)
        response = vision_model.generate_content(
            [
                Part.from_uri(gcp_url, mime_type="video/mp4"),
                prompt,
            ]
        )
        return response.text

    def parse_response(self, response_text):
        """
        Parse the response text into a JSON object.

        Args:
            response_text (str): Response text from the generative model.

        Input Format: str
        Output Format: dict (parsed JSON object)
        """
        parsed_response = parse_to_json(response_text)
        if parsed_response:
            try:
                data = json.loads(parsed_response)
                return data
            except json.JSONDecodeError as e:
                print("Failed to parse JSON:", e)
        return None

    def save_json_to_file(self, data, file_path):
        """
        Save a JSON object to a file.

        Args:
            data (dict): JSON object to save.
            file_path (str): Path to the output file.

        Input Format: dict, str
        Output Format: None
        """
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    
    def convert_time_to_seconds(self, time):
        """utility function"""
        try:
            m, s = map(int, time.split(':'))
            return m * 60 + s
        except ValueError as e:
            print(f"Error converting time to seconds: {e}")
            return None 

    def split_video_into_parts(self, video_file_path, parsed_json: dict):
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        try:
            import os
            import cv2
            # Create a directory to store the split video parts
            output_dir = os.path.splitext(video_file_path)[0]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created directory: {output_dir}")

            # Open the video file
            cap = cv2.VideoCapture(video_file_path)
            logger.info(f"Opened video file: {video_file_path}")

            # Get the video frame rate
            fps = cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Video frame rate: {fps}")

            # Iterate over the tasks in the parsed JSON
            for task, actions in parsed_json.items():
                if task not in ["overall_task_name", "objects"]:
                    for action in actions:
                        start_time = self.convert_time_to_seconds(action["start_time"])
                        end_time = self.convert_time_to_seconds(action["end_time"])

                        # Calculate the start and end frame numbers
                        start_frame = int(start_time * fps)
                        end_frame = int(end_time * fps)
                        logger.info(f"Task: {task}, Action: {action['object_name']}, Start frame: {start_frame}, End frame: {end_frame}")

                        # Create a new video writer for the split video part
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(os.path.join(output_dir, f"{task}_{action['object_name']}.mp4"), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
                        logger.info(f"Created video writer for: {task}_{action['object_name']}.mp4")

                        # Set the video capture to the start frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                        # Write the frames to the new video writer
                        for _ in range(end_frame - start_frame):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            out.write(frame)

                        # Release the video writer
                        out.release()
                        logger.info(f"Released video writer for: {task}_{action['object_name']}.mp4")

            # Release the video capture
            cap.release()
            logger.info("Released video capture")

        except Exception as e:
            logger.error(f"Error splitting video into parts: {e}")


# EXAMPLE USAGE:
def main():
    video_file_path = './videos/pick_and_place_7.mp4'
    analyzer = VideoAnalyzer(credentials_file="ai-hand-service-acc.json", project="ai-hand-423206", location="us-central1")
    gcp_url = analyzer.upload_video_to_bucket("test1.mp4", video_file_path)
    print(gcp_url)

    
    output_format= '''
    {
        overall_task_name: <a descriptive name of overall task in lower case>,
        objects: [
            <object_name>
            ...
        ]
        <action_name> : [
            {
                start_time:<start_timestamp>,
                end_time: <end_timestamp>,
                object_name: <object_name>,
                notes: <notes>,
            },
            ...
        ],
        ...
    }
    '''
    prompt = f'''the video contains a demonstration of a hunman doing a task, can you name the task and also name the objects to track in the video and also the important timestamps of what actions are happening? divide it in based on action , picking it up , placing it, therefore i want output in a format or classified label , object name and timestamps in a list note : dont focus on robotic arm , only focus on other objects and human demonistrations

    EXAMPLE OUTPUT FORMAT:
    {output_format}
    '''
    
    response_text = analyzer.generate_text("gemini-1.5-flash-002", gcp_url, prompt)
    print(response_text)

    data = analyzer.parse_response(response_text)
    analyzer.split_video_into_parts(video_file_path=video_file_path,parsed_json=data)
    if data:
        file_path = f"./outputs_json/{data['overall_task_name'].replace(' ','_')}.json"
        analyzer.save_json_to_file(data, file_path)
        print("JSON data has been saved to", file_path)

if __name__ == "__main__":
    main()
