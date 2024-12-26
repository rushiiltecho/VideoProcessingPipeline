
import json
import threading
from realsense_recording import RealSenseRecorder
from vdeo_analysis_ellm_sudio import VideoAnalyzer
from rlef_video_annotation import VideoUploader


class Application:
    
    def __init__(self, realsense_recorder:RealSenseRecorder, video_analyzer: VideoAnalyzer,rlef_annotations:VideoUploader,):
        self.realsense_recorder = realsense_recorder
        self.video_analyzer = video_analyzer
        self.rlef_annotations = rlef_annotations
        self.color_video_filepath = None
        self.is_running = False
        self.gcp_url = None

    def on_recording_stopped(self):
        """
        Callback that gets triggered automatically when the RealSenseRecorder stops recording.
        Here we simply run the application logic.
        """
        print("Application: Detected that the recording has stopped. Running post-recording steps...")
        # self.run()
        self.is_running = True


    def prepare(self,):
        """ This is a preliminary step to prepare the application for use """
        # Step 1: get the current/latest videofilepath recorded from the realsense recorder
        latest_video_path = self.realsense_recorder.get_current_savepath()
        recording_id = self.realsense_recorder.get_current_recording_id()
        # self.color_video_filepath = f'{latest_video_path}/color.mp4'
        self.color_video_filepath = f'C:\\Users\\Rushiil Bhatnagar\\Downloads\\object_detection\\object_detection\\videos\\pick_and_place_1.mp4'
        
        # Step 2: upload the video to GCP and get its URL:
        #TODO: change the name to recording_id from "test1.mp4"
        self.gcp_url = self.video_analyzer.upload_video_to_bucket(f"{'test1'}.mp4", self.color_video_filepath)

        # Step 3: get the payload for processing
        #TODO: take the gcp_url and send/set it in the payload for sending it to ELLM_STUDIO.
        payload = None
        # Load the payload from a JSON file
        with open("payload.json", "r") as file:
            payload = json.load(file)

        payload["question"] = self.gcp_url
        print(payload['question'])
        self.video_analyzer.set_payload_from_dict(payload)

    def run(self,):
        while True:
            if self.is_running:
                self.prepare()
                """ This is the main step to run the application """
                # Step 1: run the video analysis
                # TODO: store this in some kind of universal log
    # =====================================================
                response_annotations = self.video_analyzer.get_gemini_response(gcp_url=self.gcp_url)
    # =====================================================
                # response_annotations = self.video_analyzer.get_ellm_response()
    # =====================================================

                # Step 2: send these results to the annotation tool
                status = self.rlef_annotations.upload_to_rlef(url="https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/",filepath= self.color_video_filepath, video_annotations=response_annotations)
                print(f"======= status code for RLEF Upload: {status} =======")
                self.is_running = False

    def camera_loop(self,):
        """ This is a loop to continuously runs the camera and can record anytime. """
        self.realsense_recorder.capture_frames()


if __name__ == "__main__":
    realsense_recorder = RealSenseRecorder()
    video_analyzer = VideoAnalyzer()
    rlef_annotations = VideoUploader()
    app = Application(realsense_recorder, video_analyzer, rlef_annotations)
    
    # Tell the RealSenseRecorder to call app.on_recording_stopped() whenever recording is stopped
    realsense_recorder.set_recording_stopped_callback(app.on_recording_stopped)

    # Start camera in a separate thread (if you wish) or just call directly
    camera_thread = threading.Thread(target=app.camera_loop)
    camera_thread.start()
    app_run_thread = threading.Thread(target=app.run)
    app_run_thread.start()