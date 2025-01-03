import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch
from utils import get_pixel_3d_coordinates, transform_coordinates

class ObjectDepthDetector:
    def __init__(self, recording_dir, model_name="yolov8x.pt"):
        """
        Initialize detector with recording directory containing color.mp4 and frames.h5
        
        Args:
            recording_dir (str): Path to recording directory with color.mp4 and frames.h5
            model_name (str): Name of YOLO model to use (will download if not present)
        """
        self.recording_dir = Path(recording_dir)
        self.video_path = self.recording_dir / "color.mp4"
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Initialize YOLO model
        self.model = YOLO(model_name)
        
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

    def get_frame_at_time(self, time_seconds):
        """Extract frame from video at specific time"""
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
            
        return frame

    def get_object_center(self, frame, target_class):
        """
        Detect object in frame and return its center point
        
        Args:
            frame: numpy array of image
            target_class (str): Class name to detect (must be in COCO classes)
            
        Returns:
            tuple: (center_x, center_y, detection_box, confidence)
        """
        # Run detection
        results = self.model(frame)[0]
        
        # Find detections of target class
        target_detections = []
        for detection in results.boxes.data:
            class_id = int(detection[5])
            class_name = results.names[class_id]
            if class_name == target_class:
                confidence = detection[4]
                box = detection[:4]  # x1, y1, x2, y2
                target_detections.append((box, confidence))
        
        if not target_detections:
            raise ValueError(f"No {target_class} detected in frame")
            
        # Get highest confidence detection
        best_detection = max(target_detections, key=lambda x: x[1])
        box, confidence = best_detection
        
        # Calculate center point
        center_x = int((box[0] + box[2]) / 2)  # (x1 + x2) / 2
        center_y = int((box[1] + box[3]) / 2)  # (y1 + y2) / 2
        
        return center_x, center_y, box, confidence

    def get_object_3d_coordinates(self, time_seconds, target_class):
        """
        Get 3D coordinates of object center at specific time
        
        Args:
            time_seconds (float): Time in video to analyze
            target_class (str): Object class to detect (must be in COCO classes)
            
        Returns:
            dict: Dictionary containing:
                - coordinates: (X, Y, Z) coordinates in meters
                - center_pixel: (u, v) pixel coordinates
                - actual_time: actual timestamp used
                - confidence: detection confidence
                - box: detection bounding box
        """
        # Get frame at specified time
        frame = self.get_frame_at_time(time_seconds)
        
        # Detect object and get center point
        center_x, center_y, box, confidence = self.get_object_center(
            frame, target_class
        )
        
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

    def visualize_detection(self, time_seconds, target_class, output_path=None):
        """
        Visualize detection with center point and save result
        
        Args:
            time_seconds (float): Time in video
            target_class (str): Object class to detect
            output_path (str): Where to save visualization (optional)
        """
        # Get frame and detect object
        frame = self.get_frame_at_time(time_seconds)
        center_x, center_y, box, confidence = self.get_object_center(
            frame, target_class
        )
        
        # Convert box to numpy if it's a tensor
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        
        # Draw bounding box
        cv2.rectangle(
            frame,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0),
            2
        )
        
        # Draw center point
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Add text
        text = f"{target_class}: {confidence:.2f}"
        cv2.putText(
            frame, text,
            (int(box[0]), int(box[1])-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2
        )
        
        if output_path:
            cv2.imwrite(output_path, frame)
        
        return frame



# Example usage
if __name__ == "__main__":
    recording_dir = "recordings/20250102_223743"
    detector = ObjectDepthDetector(
        recording_dir,
        model_name="yolov8x-worldv2.pt"  # Using YOLOv8 extra large model
    )
    
    # Get 3D coordinates of a cup at 5 seconds
    try:
        target_class = "bottle"
        time_seconds = 4.5
        results = detector.get_object_3d_coordinates(
            time_seconds=time_seconds,
            target_class=target_class  # Must be a COCO class
        )
        
        print(f"Object detected at {results['actual_time']:.3f} seconds:")
        print(f"3D coordinates (meters):")
        print(f"X: {results['coordinates'][0]:.3f}")
        print(f"Y: {results['coordinates'][1]:.3f}")
        print(f"Z: {results['coordinates'][2]:.3f}")
        print(f"Center pixel (u,v): {results['center_pixel']}")
        print(f"Detection confidence: {results['confidence']:.2f}")
        
        # Save visualization
        detector.visualize_detection(
            time_seconds=time_seconds,
            target_class=target_class,
            output_path=f"{recording_dir}/detection_visualization.jpg"
        )

        transformed_point = transform_coordinates(
            results["coordinates"]
        )

        print(f"Transformed coordinates: {list(transformed_point)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")