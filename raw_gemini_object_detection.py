import json
import json_repair
import numpy as np
from gemini_api_key_example import GEMINI_API_KEY
import google.generativeai as genai
from PIL import Image

import io
import os
import requests

from utils import parse_list_boxes, parse_to_json, plot_bounding_boxes

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
  model_name='gemini-1.5-flash-002',
)
recording_dir = 'recordings/20250102_213303'
im = Image.open(f'{recording_dir}/detection_visualization.jpg')


prompt_ = (
            "Return bounding boxes for a soda can in the"
            " following format as a list. \n ```json{'soda_can_<color>' : [ymin, xmin, ymax,"
            " xmax], ...}``` \n If there are more than one instance of an object, add"
            " them to the dictionary as 'soda_can_<color_1>', 'soda_can_<color_2>', etc."
        )
prompt = "Return a bounding box for each of the objects in this image in \n- [ymin, xmin, ymax, xmax] \nformat."
# def get_bounding_boxes_from_gemini(im: Image, model: genai.GenerativeModel, prompt = prompt, ):
#     response = model.generate_content([
#         im,
#         prompt_
#       ])
#     print(response.text)
#     # return "DONE"
#     return json.loads(json_repair.repair_json(parse_to_json(response.text)))

# results = get_bounding_boxes_from_gemini(im, model)
# print(results)


response = model.generate_content([
    im,
    prompt_,
])
print(response.text)
boxes = json.loads(json_repair.repair_json(parse_to_json(response.text)))
# im = Image.open('detection.jpg')
boxes = {f'{i}': x for i, x in boxes.items()}
print(boxes)
plot_bounding_boxes(im, noun_phrases_and_positions=list(boxes.items()))
im.save(f'{recording_dir}/correct_detection_visualization.jpg')
print(f"DIMENSIONS: {im.width}, {im.height}")