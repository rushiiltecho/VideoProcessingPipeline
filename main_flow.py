import json
import threading
import logging
from pathlib import Path
from typing import Optional
from queue import Queue, Empty
from dataclasses import dataclass
from datetime import datetime
from realsense_recording import RealSenseRecorder
from vdeo_analysis_ellm_sudio import VideoAnalyzer
from rlef_video_annotation import VideoUploader
@dataclass
class RecordingMetadata:
    recording_id: str
    save_path: Path
    color_video_path: Path
    start_time: datetime
    end_time: Optional[datetime] = None
    frame_count: Optional[int] = None

class ApplicationState:
    def __init__(self):
        self.current_recording: Optional[RecordingMetadata] = None
        self.is_processing: bool = False
        self.processing_queue = Queue()
        self.last_processed_recording_id: Optional[str] = None

class Application:
    def __init__(self, 
                 realsense_recorder: RealSenseRecorder,
                 video_analyzer: VideoAnalyzer,
                 rlef_annotations: VideoUploader,
                 payload_template_path: str = "payload.json"):
        # Initialize logging
        self._setup_logging()
        
        # Core components
        self.realsense_recorder = realsense_recorder
        self.video_analyzer = video_analyzer
        self.rlef_annotations = rlef_annotations
        
        # State management
        self.state = ApplicationState()
        self.shutdown_flag = threading.Event()
        
        # Configuration
        self.payload_template_path = Path(payload_template_path)
        
        # Use a separate event for processing control
        self.processing_event = threading.Event()
        
        # Initialize processing thread with lower priority
        self.processing_thread = threading.Thread(
            target=self._processing_loop, 
            daemon=True,
            name="ProcessingThread"
        )
        
        # Register callbacks
        self.realsense_recorder.set_recording_stopped_callback(
            self._queue_processing_task
        )

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('application.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _queue_processing_task(self):
        """Queue processing task without blocking"""
        try:
            recording_id = self.realsense_recorder.get_current_recording_id()
            save_path = Path(self.realsense_recorder.get_current_savepath())
            
            if not recording_id or not save_path.exists():
                self.logger.error("Invalid recording metadata received")
                return
                
            metadata = RecordingMetadata(
                recording_id=recording_id,
                save_path=save_path,
                color_video_path=save_path / "color.mp4",
                start_time=datetime.now(),
                frame_count=self.realsense_recorder.frame_count
            )
            
            self.state.processing_queue.put(metadata)
            self.processing_event.set()  # Signal processing thread
            self.logger.info(f"Queued recording {recording_id} for processing")
            
        except Exception as e:
            self.logger.error(f"Error queueing processing task: {str(e)}", exc_info=True)

    def _process_recording(self, metadata: RecordingMetadata):
        """Process a single recording with resource management"""
        try:
            self.logger.info(f"Processing recording {metadata.recording_id}")
            
            # Use context managers where possible for better resource handling
            gcp_blob_name = f"{metadata.recording_id}.mp4"
            gcp_url = self.video_analyzer.upload_video_to_bucket(
                gcp_blob_name, 
                str(metadata.color_video_path)
            )
            
            with open(self.payload_template_path, 'r') as f:
                payload = json.load(f)
            payload["question"] = gcp_url
            self.video_analyzer.set_payload_from_dict(payload)
            
            annotations = self.video_analyzer.get_gemini_response(gcp_url)
            if not annotations:
                raise ValueError("Failed to get video annotations")
            
            status = self.rlef_annotations.upload_to_rlef(
                url="https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/",
                filepath=str(metadata.color_video_path),
                video_annotations=annotations
            )
            
            if status != 200:
                raise ValueError(f"RLEF upload failed with status {status}")
            
            self.state.last_processed_recording_id = metadata.recording_id
            self.logger.info(f"Successfully processed recording {metadata.recording_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing recording {metadata.recording_id}: {str(e)}", exc_info=True)
        finally:
            self.state.is_processing = False

    def _processing_loop(self):
        """Optimized processing loop with event-based waiting"""
        while not self.shutdown_flag.is_set():
            # Wait for processing signal with timeout
            if not self.processing_event.wait(timeout=1.0):
                continue
                
            try:
                while not self.shutdown_flag.is_set():
                    try:
                        # Non-blocking queue check
                        metadata = self.state.processing_queue.get_nowait()
                        self.state.is_processing = True
                        self._process_recording(metadata)
                        self.state.processing_queue.task_done()
                    except Empty:
                        # No more items to process
                        self.processing_event.clear()
                        break
                        
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}", exc_info=True)
                self.state.is_processing = False
                self.processing_event.clear()

    def start(self):
        """Start the application with optimized thread priority"""
        try:
            self.logger.info("Starting application...")
            
            # Start processing thread
            self.processing_thread.start()
            
            # Start camera recording directly
            self.realsense_recorder.capture_frames()
            
            self.logger.info("Application started successfully")
        except Exception as e:
            self.logger.error(f"Error starting application: {str(e)}", exc_info=True)
            self.stop()

    def stop(self):
        """Stop the application with graceful shutdown"""
        try:
            self.logger.info("Stopping application...")
            self.shutdown_flag.set()
            self.processing_event.set()  # Wake up processing thread
            
            # Wait for processing queue to empty with timeout
            self.state.processing_queue.join()
            
            if self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
                
            self.logger.info("Application stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping application: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    from realsense_recording import RealSenseRecorder
    from vdeo_analysis_ellm_sudio import VideoAnalyzer
    from rlef_video_annotation import VideoUploader

    realsense_recorder = RealSenseRecorder()
    video_analyzer = VideoAnalyzer()
    rlef_annotations = VideoUploader()

    app = Application(
        realsense_recorder=realsense_recorder,
        video_analyzer=video_analyzer,
        rlef_annotations=rlef_annotations
    )
    
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()