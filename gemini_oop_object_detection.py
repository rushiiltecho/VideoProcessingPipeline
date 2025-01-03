import json
from pathlib import Path
import cv2
import json_repair
import numpy as np
from PIL import Image
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional
import os
import torch 

from gemini_api_key_example import GEMINI_API_KEY
from vdeo_analysis_ellm_sudio import VideoAnalyzer
from utils import get_pixel_3d_coordinates, normalize_box, plot_bounding_boxes, transform_coordinates

class ObjectDetector:
    """A class to handle object detection using Google's Gemini model."""
    
    def __init__(self, api_key: str, model_name: str = 'gemini-1.5-flash-002', recording_dir = None):
        """
        Initialize the ObjectDetector.
        
        Args:
            api_key (str): Gemini API key
            model_name (str): Name of the Gemini model to use
        """
        self.boxes = None
        self.configure_gemini(api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.default_prompt = (
            "Return bounding boxes for a soda can in the"
            " following format as a list. \n ```json{'soda_can_<color>' : [xmin, ymin, xmax, ymax]"
            " ...}``` \n If there are more than one instance of an object, add"
            " them to the dictionary as 'soda_can_<color_1>', 'soda_can_<color_2>', etc."
        )
        self.recording_dir = Path(recording_dir)
        self.video_path = self.recording_dir / "color.mp4"

        # Open video to get properties
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        
        print(f"Video properties:")
        print(f"- Duration: {self.duration:.2f} seconds")
        print(f"- Frame count: {self.frame_count}")
        print(f"- FPS: {self.fps}")
        
        self.cap.release()

        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
    # TODO: return a PIL image
    def get_frame_at_time(self, time_seconds) -> Image.Image:
        """Extract frame from video at specific time and return as PIL image"""
        # Check if requested time is valid
        if time_seconds < 0 or time_seconds >= self.duration:
            raise ValueError(
                f"Requested time {time_seconds:.2f} seconds is outside video duration "
                f"(0 to {self.duration:.2f} seconds)"
            )
            
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")
            
        # Convert time to frame number
        frame_number = int(time_seconds * self.fps)
        
        # Ensure frame number is valid
        if frame_number >= self.frame_count:
            frame_number = self.frame_count - 1
            
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(
                f"Could not extract frame at {time_seconds} seconds "
                f"(frame {frame_number} of {self.frame_count})"
            )
        
        # Convert frame (numpy array) to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    @staticmethod
    def configure_gemini(api_key: str) -> None:
        """Configure Gemini with the provided API key."""
        genai.configure(api_key=api_key)
    
    def detect_objects(self, image_path: Optional[str]= None, image: Optional[Image.Image] = None) -> List[Dict]:
        """
        Detect objects in the given image using Gemini model.
        
        Args:
            image_path (str): Path to the image file
            prompt (str, optional): Custom prompt for the model
            
        Returns:
            Dict: Dictionary containing detected objects and their bounding boxes
        """
        try:
            im = image if image else Image.open(image_path)
            prompt_text = self.default_prompt
            
            response = self.model.generate_content([im, prompt_text])
            print(response.text)
            boxes = json.loads(json_repair.repair_json(self._parse_to_json(response.text)))
            self.boxes= boxes
            return boxes
        except Exception as e:
            print(f"EXCEPTION during detect_objects: {e}")
    
    def visualize_detections(self, 
                           image: Image.Image, 
                           boxes: Dict, 
                           output_dir: str,
                           filename: str = 'detection_visualization.jpg') -> Tuple[int, int]:
        """
        Visualize detected objects with bounding boxes and save the result.
        
        Args:
            image_path (str): Path to the original image
            boxes (Dict): Dictionary of detected objects and their bounding boxes
            output_dir (str): Directory to save the visualization
            filename (str): Name of the output file
            
        Returns:
            Tuple[int, int]: Image dimensions (width, height)
        """
        im = image # Image.open(image_path)
        self._plot_bounding_boxes(im, list(boxes.items()))
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        im.save(output_path)
        
        return im.width, im.height
    
    @staticmethod
    def _parse_to_json(text: str) -> str:
        """Extract JSON from the model's response text."""
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in the response")
        return text[start_idx:end_idx]
    
    @staticmethod
    def _plot_bounding_boxes(image: Image.Image, 
                           noun_phrases_and_positions: List[Tuple]) -> None:
        """Plot bounding boxes on the image."""
        plot_bounding_boxes(image, noun_phrases_and_positions)

    def get_real_boxes(self):
        if self.boxes is None:
            return None
        return {i: normalize_box(j) for i, j in self.boxes.items()}
    
    def get_object_center(self, im:Image, target_class):
        """
        Get the center of the detected object.
        
        Args:

            target_class (str): Object class to detect
            
        Returns:
            Tuple[int, int, np.ndarray, float]: Center coordinates, bounding box, confidence score
        """
        # Detect object
        unscaled_boxes = self.detect_objects(image=im) 
        boxes = self.get_real_boxes()
        self.visualize_detections(im, unscaled_boxes, self.recording_dir)
        if target_class not in boxes:
            return None, None, None, None
        
        # Get bounding box and confidence score
        box = boxes[target_class]
        confidence = 100
        
        # Calculate center coordinates
        center_x = int((box[0] + box[2]) / 2)
        center_y = int((box[1] + box[3]) / 2)
        
        return center_x, center_y, box, confidence
    
    def get_object_3d_coordinates(self, time_seconds, target_class):
        """
        Get 3D coordinates of object center at specific time
        
        Args:
            time_seconds (float): Time in video to analyze
            target_class (str): Object class to detect (must be in COCO classes)
            
        Returns:
            dict: Dictionary containing:
                - coordinates: (Z , Y , X) coordinates in meters
                - center_pixel: (u, v) pixel coordinates
                - actual_time: actual timestamp used
                - confidence: detection confidence
                - box: detection bounding box
        """
        # Get frame at specified time
        frame = self.get_frame_at_time(time_seconds)
        
        # Detect object and get center point
        center_x, center_y, box, confidence = self.get_object_center(im=frame, target_class=target_class)
        
        # Get 3D coordinates of center point
        coords, actual_time = get_pixel_3d_coordinates(
            self.recording_dir,
            time_seconds,
            center_x,
            center_y
        )
        
        return {
            "coordinates": coords,
            "center_pixel": (center_x, center_y),
            "actual_time": actual_time,
            "confidence": confidence,
            "box": box.cpu().numpy() if isinstance(box, torch.Tensor) else box
        }

def ellm_studio_test(recording_dir:str):
    payload = None

    # Load the payload from a JSON file
    with open("payload.json", "r") as file:
        payload = json.load(file)

    print(f"================ PAYLOAD ================ +\n{payload['question']}\n================ PAYLOAD ================")
    analyzer = VideoAnalyzer(payload=payload)
    # gcp_url = analyzer.upload_video_to_bucket("test1.mp4", f'{recording_dir}/color.mp4')
    response = analyzer.get_gemini_response()
    # print(response)
    return response

# Example usage:
if __name__ == "__main__":
    recording_dir = 'recordings/20241225_140621'
    response = ellm_studio_test(recording_dir=recording_dir)
    # print(response)
    classes_to_detect = response['objects']
    # TODO: Modify Prompt: get actions in agent response as a separate field to use differently, just like objects
    actions = list(response.keys())[2:]
    # print(actions)
    # Initialize detector
    detector = ObjectDetector(api_key=GEMINI_API_KEY, recording_dir= recording_dir)

    # Get 3D coordinates
    time_seconds = 1
    target_class = 'soda_can_red'
    result = detector.get_object_3d_coordinates(time_seconds, target_class)
    print(f"3D coordinates at {time_seconds} seconds:", result)

    print(f'\n\n\n CORRECTED COORDINATES: {transform_coordinates(result["coordinates"])}')