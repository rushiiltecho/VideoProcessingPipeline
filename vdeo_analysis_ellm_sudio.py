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
from typing import Dict, List, Optional
import json_repair
import requests
import vertexai
from google.cloud import storage
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import os
from utils import parse_to_json
import cv2

class VideoAnalyzer:
    def __init__(
                 self,
                 payload_filepath:Optional[str]=None, 
                 payload: Optional[dict]=None,
                 egpt_url:str="https://dev-egpt.techo.camp/predict", 
                 headers:dict={"Content-Type": "application/json"}, 
                 google_credential_filepath:str='ai-hand-service-acc.json', 
                 project:str='ai-hand-423206', 
                 location:str='us-central1'
                ):
        """
        Initialize the VideoAnalyzer class.

        Args:
            payload (Optional[dict]) : payload dictionary
            payload_filepath (Optional[str]): Path to the payload file
            (Note: give either payload or payload_filepath)
            credentials_file (str): Path to the Google application credentials file.
            project (str): Google Cloud project ID.
            location (str): Google Cloud location.

        Input Format: None
        Output Format: None
        """        
        self.credentials_file = google_credential_filepath
        self.project = project
        self.location = location
        self.egpt_url = egpt_url
        self.headers = headers
        if payload_filepath:
            self.payload_filepath = payload_filepath
            with open("payload.json", "r") as file:
                payload = json.load(file) 
            self.payload = payload
        if payload:
            self.payload = payload if payload else None
        try:
            if payload or payload_filepath:
                pass
            else: 
                Exception ("Either payload or payload_filepath must be provided")
        except Exception as e:
            print(f"Error: {e}")
        self.credentials = service_account.Credentials.from_service_account_file(google_credential_filepath)
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

    def set_payload_from_dict(self,payload):
        self.payload = payload

    def set_payload_from_filepath(self, payload_filepath):
        self.payload_filepath = payload_filepath
        payload = None
        with open(payload_filepath, "r") as file:
            payload = json.load(file)
        
        self.payload = payload

    def get_ellm_response(self):
        response = requests.post(self.egpt_url, headers=self.headers, data=json.dumps(self.payload))
        # print(response.text)
        processed_response = self.process_ellm_response(response.text)
        print(type(processed_response))
        # print(f"RAW RESPONSE: {processed_response}")
        return processed_response

    def process_ellm_response(self, response):
        response_content:str= None
        try:
            response = json.loads(response)
            # print(response)
            # print(type(response))

            # Step 1: Extract Chat History array's element which has tool_calls key
            # Step 2: Get the "id"key where "tool_calls" key has a sub-key ["tool_calls"][i]["function"]["name"] having value: "AnalyzeVideoAgent" or self.payload['agents'][i]['title'] == <AGENT_NAME>
            # Step 3: Get the element in chat_history where "role" is "tool" and "tool_call_id" is the extracted tool call id.
            # Step 4: Extract "content" key from that element of the chat history.
            if response:
                chat_history:List[Dict] = response['chatHistory']
                tool_call_id = None
                response_content = None
                tool_call_idx, chat_history_idx = None, None
                for i in range(len(chat_history)):
                    if 'tool_calls' in chat_history[i].keys():
                        for j in range(len(chat_history[i]['tool_calls'])):
                            if chat_history[i]['tool_calls'][j]['function']['name'] == 'AnalyzeVideoAgent':
                                tool_call_id = chat_history[i]['tool_calls'][j]['id']
                                chat_history_idx, tool_call_idx = i, j
                                break
                for i in range(len(chat_history)):
                    if chat_history[i]["role"] == "tool" and chat_history[i]["tool_call_id"] == tool_call_id :
                        response_content= chat_history[i]["content"]
                        break

                print("response_content", response_content)

        except:
            pass
        finally:
            if response_content:
                to_parse = json.loads(json_repair.repair_json(response_content))
                parsed = self.parse_response(to_parse['content'])
                # print(parsed['content'])
                return parsed
            return None

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
        print(f"response_text: {response_text}")
        parsed_response = parse_to_json(response_text)
        print(f"parsed_response: {parsed_response}")
        if parsed_response:
            try:
                data = json.loads(parsed_response)
                print("DATA ========== \n", data)
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
    
    def get_gemini_response(self, gcp_url):
        vision_model_name = "gemini-1.5-flash-002"
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
        response = self.generate_text(vision_model_name=vision_model_name,gcp_url=gcp_url,prompt=prompt)
        parsed_data = self.parse_response(response_text=response)

        print(parsed_data)
        return parsed_data

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
    response = analyzer.get_ellm_response()
    
    
    # main()
